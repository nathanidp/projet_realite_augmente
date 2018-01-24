using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using System;
using System.Drawing;
using System.Windows.Forms;
using Emgu.Util;
using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.Face;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;


public class ChessboardDetection : MonoBehaviour
{
	public static WebCamTexture webcamTexture;
	private static Texture2D displayTexture;
	private Color32[] data;
	private byte[] bytes;
	private byte[] grayBytes;
	private FlipType flip = FlipType.Horizontal;
	private Size patternSize = new Size(7, 4);
	private MCvTermCriteria criteria = new MCvTermCriteria(100, 1e-5);
	public GameObject screen;
	public GameObject virtual_chessboard;

	void Start()
	{
		WebCamDevice[] devices = WebCamTexture.devices;
		int cameraCount = devices.Length;

		if (cameraCount > 0)
		{
			webcamTexture = new WebCamTexture(devices[0].name);
			webcamTexture.Play();
		}
	}

	void Update()
	{
		if (webcamTexture != null && webcamTexture.didUpdateThisFrame)
		{
			if (data == null || (data.Length != webcamTexture.width * webcamTexture.height))
				data = new Color32[webcamTexture.width * webcamTexture.height];

			webcamTexture.GetPixels32(data);
			//data = webcamTexture.GetPixels32(0);

			if (bytes == null || bytes.Length != data.Length * 3)
				bytes = new byte[data.Length * 3];
			if (grayBytes == null || grayBytes.Length != data.Length * 1)
				grayBytes = new byte[data.Length * 1];

			// OPENCV PROCESSING
			GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
			GCHandle resultHandle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
			GCHandle grayHandle = GCHandle.Alloc(grayBytes, GCHandleType.Pinned);

			Mat currentWebcamMat = new Mat(new Size(webcamTexture.width, webcamTexture.height), DepthType.Cv8U, 4, handle.AddrOfPinnedObject(), webcamTexture.width * 4);
			Mat resultMat = new Mat(webcamTexture.height, webcamTexture.width, DepthType.Cv8U, 3, resultHandle.AddrOfPinnedObject(), webcamTexture.width * 3);
			Mat grayMat = new Mat(webcamTexture.height, webcamTexture.width, DepthType.Cv8U, 1, grayHandle.AddrOfPinnedObject(), webcamTexture.width * 1);

			CvInvoke.CvtColor(currentWebcamMat, resultMat, ColorConversion.Bgra2Bgr);
			CvInvoke.CvtColor(resultMat, grayMat, ColorConversion.Bgra2Gray);
			//VectorOfPoint cornerPoints = DetectCheckerboard(grayMat);


			if (Input.GetKey(KeyCode.D))
				FindCameraProperties(grayMat, resultMat); //DetectCheckerboard(grayMat, resultMat); 


			handle.Free();
			resultHandle.Free();
			grayHandle.Free();

			if (flip != FlipType.None)
				CvInvoke.Flip(resultMat, resultMat, flip);
			if (displayTexture == null || displayTexture.width != webcamTexture.width ||
				displayTexture.height != webcamTexture.height)
			{
				displayTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
			}
			displayTexture.LoadRawTextureData(bytes);
			displayTexture.Apply();
		}

		if (displayTexture != null)
		{
			screen.GetComponent<MeshRenderer>().sharedMaterial.SetTexture("_MainTex", displayTexture);
		}
	}

	private void DetectCheckerboard(Mat detectImage, Mat drawImage)
	{
		Matrix<float> cornerPoints = new Matrix<float>(patternSize);
		bool result = CvInvoke.FindChessboardCorners(detectImage, patternSize, cornerPoints);

		if (result == false)
			return;
		CvInvoke.DrawChessboardCorners(drawImage, patternSize, cornerPoints, true);
	}

	void SetCameraProperties(Matrix<float> calibration, Matrix<float> rotation, Matrix<float> translation, Matrix4x4 projection, Matrix4x4 modelview)
	{
		float zNear = Camera.main.nearClipPlane;
		float zFar = Camera.main.farClipPlane;

		projection.m00 = 2 * calibration[0, 0] / 640;
		projection.m10 = 0;
		projection.m20 = 0;
		projection.m30 = 0;

		projection.m01 = 0;
		projection.m11 = 2 * calibration[1, 1] / 480;
		projection.m21 = 0;
		projection.m31 = 0;

		projection.m02 = 1 - 2 * calibration[0, 2] / 640;
		projection.m12 = -1 + (2 * calibration[1, 2] + 2) / 480;
		projection.m22 = (zNear + zFar) / (zNear - zFar);
		projection.m32 = -1;

		projection.m03 = 0;
		projection.m13 = 0;
		projection.m23 = 2 * zNear * zFar / (zNear - zFar);
		projection.m33 = 0;


		modelview.m00 = rotation[0, 0];
		modelview.m10 = rotation[1, 0];
		modelview.m20 = rotation[2, 0];
		modelview.m30 = 0;

		modelview.m01 = rotation[0, 1];
		modelview.m11 = rotation[1, 1];
		modelview.m21 = rotation[2, 1];
		modelview.m31 = 0;

		modelview.m02 = rotation[0, 2];
		modelview.m12 = rotation[1, 2];
		modelview.m22 = rotation[2, 2];
		modelview.m32 = 0;

		modelview.m03 = translation[0, 0];
		modelview.m13 = translation[1, 0];
		modelview.m23 = translation[2, 0];
		modelview.m33 = 1;

		// This matrix corresponds to the change of coordinate systems.
		///static double changeCoordArray[4][4] = {{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}};
		//static Mat changeCoord(4, 4, CV_64FC1, changeCoordArray);
		//modelview = changeCoord* modelview;

		Vector3 t = ExtractTranslation(modelview);
		Vector3 s = ExtractScale(modelview);
		Quaternion r = ExtractRotation(modelview, s);

		Camera.main.transform.position = t;
		Camera.main.transform.localScale = s;
		Camera.main.transform.rotation = r;
	}

	void FindCameraProperties(Mat image, Mat result)
	{
		Matrix<float> realCorners = new Matrix<float>(patternSize);

		// Try to find the chess board corners in the image.
		bool foundCorners = CvInvoke.FindChessboardCorners(image, patternSize, realCorners);

		// If we weren't able to find the corners exit early.
		if (!foundCorners)
			return;

		CvInvoke.DrawChessboardCorners(result, patternSize, realCorners, true);

		Matrix4x4 modelview = new Matrix4x4();
		Matrix4x4 projection = new Matrix4x4();
		VectorOfPoint3D32F virtualCorners = new VectorOfPoint3D32F(); // The corresponding corner positions for where the corners lie on the chess board (measured in virtual units).
		Mat rotation = new Mat(4, 4, DepthType.Default, 1);        // The calculated rotation of the chess board.
		Matrix<float> translation = new Matrix<float>(4, 4);  // The calculated translation of the chess board.

		BuildVirtualCorners(virtualCorners, 2);

		Matrix<float> calibrationMatrix = new Matrix<float>(new Size(3, 3));
		Matrix<float> calibrationDistortionCoefficients = new Matrix<float>(new Size(1, 5));

		//   1.2306403943428504e+03 0. 960. 
		//   0. 1.2306403943428504e+03 540. 
		//   0. 0. 1.
		calibrationMatrix[0, 0] = 1230;
		calibrationMatrix[0, 1] = 0.0f;
		calibrationMatrix[0, 2] = 960;

		calibrationMatrix[1, 0] = 0.0f;
		calibrationMatrix[1, 1] = 1230;
		calibrationMatrix[1, 2] = 540;

		calibrationMatrix[2, 0] = 0.0f;
		calibrationMatrix[2, 1] = 0.0f;
		calibrationMatrix[2, 2] = 1;

		// 1.9920531921963049e-02 3.2143454945024297e-02 0. 0. -2.2585645769105978e-01
		calibrationDistortionCoefficients[0, 0] = 0.0199f;
		calibrationDistortionCoefficients[1, 0] = 0.032143f;
		calibrationDistortionCoefficients[2, 0] = 0.0f;
		calibrationDistortionCoefficients[3, 0] = 0.0f;
		calibrationDistortionCoefficients[4, 0] = -0.22585f;

		// Compute the rotation / translation of the chessboard (the cameras extrinsic pramaters).
		CvInvoke.SolvePnP(virtualCorners, realCorners, calibrationMatrix, calibrationDistortionCoefficients, rotation, translation);

		// Converte the rotation from 3 axis rotations into a rotation matrix.
		Matrix<float> rotationMatrix = new Matrix<float>(4, 4);
		CvInvoke.Rodrigues(rotation, rotationMatrix);

		// The tranlation corresponds to the origin, which is at the corner of the chess board
		// but I would like to define the origin so that it is at the center of the chess board
		// so I need to offset by half of the size of the chessboard and need to multiply it by
		// the rotation so that it is in the local coordinate system of the chessboard.
		// double offsetA[3][1] = {{(chessCornersX-1.0)/2.0}, {(chessCornersY-1.0)/2.0}, {0}};
		// Mat offset = new Mat(3, 1, DepthType.Cv64F, 1);

		Matrix<float> t = new Matrix<float>(4, 4);
		CvInvoke.Add(translation, rotation, t);
		translation = t;

		// Turn the intrinsic and extrinsic pramaters into the projection and modelview matrix for OpenGL to use.
		SetCameraProperties(calibrationMatrix, rotationMatrix, translation, projection, modelview);

		//return ai;
	}

	/** En français : on init les coordonnées des corners du chestboard virtuels
    * @brief Constructs a matrix which holds all of the corner points of the chessboard.
    * @param corners The list of all the corners.
    * @param scale The scale of the chess board corners (i.e. 1 square = how many units).
    *
    * The corners are ordered column at at time (as opposed to row at at time),
    * ie. column 1 then column 2 etc.
    */
	void BuildVirtualCorners(VectorOfPoint3D32F corners, float scale)
	{
		MCvPoint3D32f[] tmp = new MCvPoint3D32f[patternSize.Height * patternSize.Width];
		if (corners != null)
		{
			for (int ix = 0; ix < patternSize.Width; ix++)
			{
				for (int iy = 0; iy < patternSize.Height; iy++)
					tmp[iy * patternSize.Width + ix] = new MCvPoint3D32f(iy * scale, ix * scale, 0);
			}
		}
		corners.Push(tmp);
	}


	private Vector3 ExtractScale(Matrix4x4 modelView)
	{
		Vector3 a = new Vector3(modelView.m00, modelView.m10, modelView.m20);
		Vector3 b = new Vector3(modelView.m10, modelView.m11, modelView.m12);
		Vector3 c = new Vector3(modelView.m20, modelView.m21, modelView.m22);
		return new Vector3(a.magnitude, b.magnitude, c.magnitude);
	}

	private Quaternion ExtractRotation(Matrix4x4 modelView, Vector3 scale)
	{
		modelView.m00 /= scale.x;
		modelView.m01 /= scale.x;
		modelView.m02 /= scale.x;
		modelView.m03 = 0;

		modelView.m10 /= scale.x;
		modelView.m11 /= scale.y;
		modelView.m12 /= scale.y;
		modelView.m13 = 0;

		modelView.m20 /= scale.z;
		modelView.m21 /= scale.z;
		modelView.m22 /= scale.z;
		modelView.m23 = 0;

		modelView.m30 = 0;
		modelView.m31 = 0;
		modelView.m32 = 0;
		modelView.m33 = 1;

		return Quaternion.LookRotation(modelView.GetColumn(2), modelView.GetColumn(1));
	}

	private Vector3 ExtractTranslation(Matrix4x4 modelView)
	{
		return new Vector3(modelView.m30, modelView.m31, modelView.m32);
	}
}
