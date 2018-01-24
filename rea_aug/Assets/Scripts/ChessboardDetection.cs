using System.Runtime.InteropServices;
using UnityEngine;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

public class ChessboardDetection : MonoBehaviour
{
	public static WebCamTexture webcamTexture;
	private static Texture2D displayTexture;

	public GameObject screen;
	public GameObject virtual_chessboard;

	private FlipType flip = FlipType.Horizontal;
	private Size patternSize = new Size(7, 4);

	private Color32[] data;
	private byte[] bytes;
	private byte[] grayBytes;

	Matrix<float> cornerPointImage;
	Matrix<float> virtualCorners;

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

			if (Input.GetKey(KeyCode.D))
				FindCameraProperties(grayMat, resultMat); //FindCameraProperties(grayMat, resultMat);

			handle.Free();
			resultHandle.Free();
			grayHandle.Free();

			if (flip != FlipType.None)
				CvInvoke.Flip(resultMat, resultMat, flip);
			if (displayTexture == null || displayTexture.width != webcamTexture.width || displayTexture.height != webcamTexture.height)
				displayTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
			displayTexture.LoadRawTextureData(bytes);
			displayTexture.Apply();
		}

		if (displayTexture != null)
			screen.GetComponent<MeshRenderer>().sharedMaterial.SetTexture("_MainTex", displayTexture);
	}

	private bool FindChessboardCorners(Mat detectImage, Mat drawImage)
	{
		cornerPointImage = new Matrix<float>(patternSize.Width * patternSize.Height, 1, 2);
		bool found = CvInvoke.FindChessboardCorners(detectImage, patternSize, cornerPointImage);
		if (found)
			CvInvoke.DrawChessboardCorners(drawImage, patternSize, cornerPointImage, found);
		return found;
	}

	private void BuildVirtualCorners(float scale)
	{
		for (int iy = 0; iy < patternSize.Height; iy++)
		{
			for (int ix = 0; ix < patternSize.Width; ix++)
			{
				int index = iy * patternSize.Width + ix;
				virtualCorners.Data[index, 0] = iy * scale;
				virtualCorners.Data[index, 1] = ix * scale;
				virtualCorners.Data[index, 2] = 0;
			}
		}
	}

	private void GetCalibrationMatrices(Matrix<float> calibrationMatrix, Matrix<float> calibrationDistortionCoefficients)
	{
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
		calibrationDistortionCoefficients[0, 1] = 0.032143f;
		calibrationDistortionCoefficients[0, 2] = 0.0f;
		calibrationDistortionCoefficients[0, 3] = 0.0f;
		calibrationDistortionCoefficients[0, 4] = -0.22585f;
	}

	private void SetCameraProperties(Matrix<float> calibration, float[] rotation, float[] translation)
	{
		Matrix4x4 modelview = new Matrix4x4();
		Matrix4x4 projection = new Matrix4x4();

		float zNear = Camera.main.nearClipPlane;
		float zFar = Camera.main.farClipPlane;

		projection.m00 = 2 * calibration[0, 0] / 960;
		projection.m10 = 0;
		projection.m20 = 0;
		projection.m30 = 0;

		projection.m01 = 0;
		projection.m11 = 2 * calibration[1, 1] / 540;
		projection.m21 = 0;
		projection.m31 = 0;

		projection.m02 = 1 - 2 * calibration[0, 2] / 960;
		projection.m12 = -1 + (2 * calibration[1, 2] + 2) / 540;
		projection.m22 = (zNear + zFar) / (zNear - zFar);
		projection.m32 = -1;

		projection.m03 = 0;
		projection.m13 = 0;
		projection.m23 = 2 * zNear * zFar / (zNear - zFar);
		projection.m33 = 0;


		modelview.m00 = rotation[0 * 3];
		modelview.m10 = rotation[1 * 3];
		modelview.m20 = rotation[2 * 3];
		modelview.m30 = 0;

		modelview.m01 = rotation[0 * 3 + 1];
		modelview.m11 = rotation[1 * 3 + 1];
		modelview.m21 = rotation[2 * 3 + 1];
		modelview.m31 = 0;

		modelview.m02 = rotation[0 * 3 + 2];
		modelview.m12 = rotation[1 * 3 + 2];
		modelview.m22 = rotation[2 * 3 + 2];
		modelview.m32 = 0;

		modelview.m03 = translation[0];
		modelview.m13 = translation[2];
		modelview.m23 = translation[1];
		modelview.m33 = 1;

		Vector3 t = ExtractTranslation(modelview);
		Debug.Log(t);
		//Quaternion r = ExtractRotation(modelview);

		//Camera.main.transform.rotation = r;
		Camera.main.transform.position = t;
		Camera.main.projectionMatrix = projection;
	}

	private void FindCameraProperties(Mat image, Mat result)
	{
		bool foundCorners = FindChessboardCorners(image, result);
		if (!foundCorners)
			return;

		// Virtual Corners
		virtualCorners = new Matrix<float>(patternSize.Height * patternSize.Width, 1, 3);
		BuildVirtualCorners(1);
		
		// Calibration matrices : Intrinsec parameters
		Matrix<float> calibrationMat = new Matrix<float>(new Size(3, 3));
		Matrix<float> distortionCoefs = new Matrix<float>(new Size(5, 1));
		GetCalibrationMatrices(calibrationMat, distortionCoefs);

		// Rotation & Translation
		Mat rotVector = new Mat(3, 1, DepthType.Cv64F, 1);
		Mat transVector = new Mat(3, 1, DepthType.Cv64F, 1);

		// Compute the rotation / translation of the chessboard (the cameras extrinsic pramaters).
		if (CvInvoke.SolvePnP(virtualCorners, cornerPointImage, calibrationMat, distortionCoefs, rotVector, transVector) == false)
			Debug.Log("SolvePNP : false");

		// Converte the rotation from 3 axis rotations into a rotation matrix.
		Mat rotationMatrix = new Mat(3, 3, DepthType.Cv64F, 1);
		CvInvoke.Rodrigues(rotVector, rotationMatrix);

		float[] rotationData = new float[9];
		Marshal.Copy(rotationMatrix.DataPointer, rotationData, 0, rotationMatrix.Width * rotationMatrix.Height);
		float[] translationData = new float[3];
		Marshal.Copy(transVector.DataPointer, translationData, 0, transVector.Width * transVector.Height);

		// Turn the intrinsic and extrinsic pramaters into the projection and modelview matrix for OpenGL to use.
		SetCameraProperties(calibrationMat, rotationData, translationData);
	}

	private Vector3 ExtractScale(Matrix4x4 modelView)
	{
		Vector3 a = new Vector3(modelView.m00, modelView.m10, modelView.m20);
		Vector3 b = new Vector3(modelView.m10, modelView.m11, modelView.m12);
		Vector3 c = new Vector3(modelView.m20, modelView.m21, modelView.m22);
		return new Vector3(a.magnitude, b.magnitude, c.magnitude);
	}

	private Quaternion ExtractRotation(Matrix4x4 modelView)
	{
		Vector3 forward;
		forward.x = modelView.m02;
		forward.y = modelView.m12;
		forward.z = modelView.m22;

		Vector3 up;
		up.x = modelView.m01;
		up.y = modelView.m11;
		up.z = modelView.m21;

		return Quaternion.LookRotation(forward, up);
	}

	private Vector3 ExtractTranslation(Matrix4x4 modelView)
	{
		Vector3 position;
		position.x = modelView.m03;
		position.y = modelView.m13;
		position.z = modelView.m23;
		return position;
	}
}
