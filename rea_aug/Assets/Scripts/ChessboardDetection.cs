using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;


[System.Serializable]
public class CamExtrinsicParam
{
	public double[] rotationData;
	public double[] translationData;
}

public class ChessboardDetection : MonoBehaviour
{
	private static WebCamTexture webcamTexture;
	private static Texture2D displayTexture;

	[Header("Parameters")]
	public RawImage canvasImageDisplay;
	public GameObject virtualChessboard;
	public int patternWidth = 7;
	public int patternHeight = 4;

	// All computation matrices
	private Matrix<float> imageCorners;
	private Matrix<double> virtualCorners;
	private Matrix<double> intrinsicMatrix;
	private Matrix<double> distortionMatrix;

	// Intermediary Matrices & Handles & Data arrays
	private Mat currentWebcamMat;
	private Color32[] currentWebcamData;
	private GCHandle currentWebcamHandle;

	private Mat webcamModifiedMat;
	private byte[] webcamModifedData;
	private GCHandle webcamModifiedHandle;

	private Mat webcamGrayMat;
	private byte[] webcamGrayData;
	private GCHandle webcamGrayHandle;


	private void Start()
	{
		// Get webcam device & display
		WebCamDevice[] devices = WebCamTexture.devices;
		int cameraCount = devices.Length;
		if (cameraCount > 0)
		{
			webcamTexture = new WebCamTexture(devices[0].name);
			webcamTexture.Play();
		}

		// Construct virtual corner points
		Vector2 offset = new Vector2(patternWidth / 2.0f, patternHeight / 2.0f);
		virtualCorners = new Matrix<double>(patternHeight * patternWidth, 1, 3);
		for (int iy = 0; iy < patternHeight; iy++)
		{
			for (int ix = 0; ix < patternWidth; ix++)
			{
				virtualCorners.Data[iy * patternWidth + ix, 0] = ix - offset.x;
				virtualCorners.Data[iy * patternWidth + ix, 1] = iy - offset.y;
				virtualCorners.Data[iy * patternWidth + ix, 2] = 0;
			}
		}

		// Initialize intrinsic parameters
		intrinsicMatrix = new Matrix<double>(3, 3, 1);
		intrinsicMatrix[0, 0] = 1.2306403943428504e+03f;
		intrinsicMatrix[0, 1] = 0;
		intrinsicMatrix[0, 2] = webcamTexture.width / 2.0d;
		intrinsicMatrix[1, 0] = 0;
		intrinsicMatrix[1, 1] = 1.2306403943428504e+03f;
		intrinsicMatrix[1, 2] = webcamTexture.height / 2.0d;
		intrinsicMatrix[2, 0] = 0;
		intrinsicMatrix[2, 1] = 0;
		intrinsicMatrix[2, 2] = 1;

		distortionMatrix = new Matrix<double>(4, 1, 1);
		distortionMatrix[0, 0] = 1.9920531921963049e-02f;
		distortionMatrix[1, 0] = 3.2143454945024297e-02f;
		distortionMatrix[2, 0] = 0.0f;
		distortionMatrix[3, 0] = 0.0f;
	}

	private void Update()
	{
		if (webcamTexture != null && webcamTexture.didUpdateThisFrame)
		{
			// Update new data from webcam device
			UpdateWebcamBegin();

			// Detect & Draw chessboard corners
			bool detected = FindAndDrawChessboardCorners(webcamGrayMat, webcamModifiedMat);
			if (detected == true && Input.GetKey(KeyCode.D))
			{
				// If detected, update camera transform accordingly
				CamExtrinsicParam camParams = GetExtrinsicParameters();
				UpdateCameraTransform(camParams);
			}

			// Apply modified data to webcam (image with corners detected)
			UpdateWebcamEnd();
		}

		if (displayTexture != null)
			canvasImageDisplay.texture = displayTexture;
	}


	/* Webcam Functions */
	private void UpdateWebcamBegin()
	{
		if (currentWebcamData == null || (currentWebcamData.Length != webcamTexture.width * webcamTexture.height))
			currentWebcamData = new Color32[webcamTexture.width * webcamTexture.height];
		webcamTexture.GetPixels32(currentWebcamData);
		if (webcamModifedData == null || webcamModifedData.Length != currentWebcamData.Length * 3)
			webcamModifedData = new byte[currentWebcamData.Length * 3];
		if (webcamGrayData == null || webcamGrayData.Length != currentWebcamData.Length * 1)
			webcamGrayData = new byte[currentWebcamData.Length * 1];

		currentWebcamHandle = GCHandle.Alloc(currentWebcamData, GCHandleType.Pinned);
		webcamModifiedHandle = GCHandle.Alloc(webcamModifedData, GCHandleType.Pinned);
		webcamGrayHandle = GCHandle.Alloc(webcamGrayData, GCHandleType.Pinned);

		currentWebcamMat = new Mat(new Size(webcamTexture.width, webcamTexture.height), DepthType.Cv8U, 4, currentWebcamHandle.AddrOfPinnedObject(), webcamTexture.width * 4);
		webcamModifiedMat = new Mat(webcamTexture.height, webcamTexture.width, DepthType.Cv8U, 3, webcamModifiedHandle.AddrOfPinnedObject(), webcamTexture.width * 3);
		webcamGrayMat = new Mat(webcamTexture.height, webcamTexture.width, DepthType.Cv8U, 1, webcamGrayHandle.AddrOfPinnedObject(), webcamTexture.width * 1);

		CvInvoke.CvtColor(currentWebcamMat, webcamModifiedMat, ColorConversion.Bgra2Bgr);
		CvInvoke.CvtColor(webcamModifiedMat, webcamGrayMat, ColorConversion.Bgra2Gray);
	}

	private void UpdateWebcamEnd()
	{
		currentWebcamHandle.Free();
		webcamModifiedHandle.Free();
		webcamGrayHandle.Free();

		if (displayTexture == null || displayTexture.width != webcamTexture.width || displayTexture.height != webcamTexture.height)
			displayTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
		displayTexture.LoadRawTextureData(webcamModifedData);
		displayTexture.Apply();
	}


	/* Camera Functions */
	private bool FindAndDrawChessboardCorners(Mat detectImage, Mat drawImage)
	{
		imageCorners = new Matrix<float>(patternWidth * patternHeight, 1, 2);
		bool found = CvInvoke.FindChessboardCorners(detectImage, new Size(patternWidth, patternHeight), imageCorners);
		if (found)
			CvInvoke.DrawChessboardCorners(drawImage, new Size(patternWidth, patternHeight), imageCorners, found);
		return found;
	}

	private CamExtrinsicParam GetExtrinsicParameters()
	{
		// Compute camera extrinsic parameters : rotation & translation matrices.
		Mat rotationMat = new Mat(3, 1, DepthType.Cv64F, 1);
		Mat translationMat = new Mat(3, 1, DepthType.Cv64F, 1);
		bool res = CvInvoke.SolvePnP(virtualCorners, imageCorners, intrinsicMatrix, distortionMatrix, rotationMat, translationMat);
		if (res == false)
		{
			Debug.Log("SolvePnP Failed!");
			return null;
		}

		// Convert the rotation from 3 axis rotations into a rotation matrix.
		double[] tempRotationData = new double[3];
		Marshal.Copy(rotationMat.DataPointer, tempRotationData, 0, rotationMat.Width * rotationMat.Height);
		tempRotationData[0] = tempRotationData[0];
		tempRotationData[1] = -tempRotationData[1];
		tempRotationData[2] = -tempRotationData[2];
		GCHandle tempHandle = GCHandle.Alloc(tempRotationData, GCHandleType.Pinned);

		// Final rotation matrix with Rodrigues function.
		rotationMat = new Mat(3, 1, DepthType.Cv64F, 1, tempHandle.AddrOfPinnedObject(), sizeof(double));
		Mat finalRotationMat = new Mat(3, 3, DepthType.Cv64F, 1);
		CvInvoke.Rodrigues(rotationMat, finalRotationMat);


		// Get all data to set Camera transform.
		double[] rotationData = new double[9];
		Marshal.Copy(finalRotationMat.DataPointer, rotationData, 0, finalRotationMat.Width * finalRotationMat.Height);
		double[] translationData = new double[3];
		Marshal.Copy(translationMat.DataPointer, translationData, 0, translationMat.Width * translationMat.Height);


		// Return data.
		CamExtrinsicParam ret = new CamExtrinsicParam();
		ret.rotationData = rotationData;
		ret.translationData = translationData;
		return ret;
	}

	private void UpdateCameraTransform(CamExtrinsicParam camParams)
	{
		double[] extrinsicRotation = camParams.rotationData;
		double[] extrinsicTranslat = camParams.translationData;


		// Projection matrix
		Matrix4x4 projection = new Matrix4x4();
		projection.m00 = (float)(2 * intrinsicMatrix[0, 0] / webcamTexture.width);
		projection.m10 = 0;
		projection.m20 = 0;
		projection.m30 = 0;

		projection.m01 = 0;
		projection.m11 = (float)(2 * intrinsicMatrix[1, 1] / webcamTexture.height);
		projection.m21 = 0;
		projection.m31 = 0;

		projection.m02 = (float)(1 - 2 * intrinsicMatrix[0, 2] / webcamTexture.width);
		projection.m12 = (float)(-1 + (2 * intrinsicMatrix[1, 2] + 2) / webcamTexture.height);
		projection.m22 = (Camera.main.nearClipPlane + Camera.main.farClipPlane) / (Camera.main.nearClipPlane - Camera.main.farClipPlane);
		projection.m32 = -1;

		projection.m03 = 0;
		projection.m13 = 0;
		projection.m23 = 2 * (Camera.main.nearClipPlane * Camera.main.farClipPlane) / (Camera.main.nearClipPlane - Camera.main.farClipPlane);
		projection.m33 = 0;

		Camera.main.projectionMatrix = projection;
		Camera.main.fieldOfView = Mathf.Atan(1.0f / projection.m11) * 2.0f * Mathf.Rad2Deg;
		Camera.main.aspect = webcamTexture.width / webcamTexture.height;


		// Model view matrix
		Matrix4x4 modelViewMatrix = new Matrix4x4();
		modelViewMatrix.m00 = (float)extrinsicRotation[0];
		modelViewMatrix.m10 = (float)extrinsicRotation[3];
		modelViewMatrix.m20 = (float)extrinsicRotation[6];
		modelViewMatrix.m30 = 0;

		modelViewMatrix.m01 = (float)extrinsicRotation[1];
		modelViewMatrix.m11 = (float)extrinsicRotation[4];
		modelViewMatrix.m21 = (float)extrinsicRotation[7];
		modelViewMatrix.m31 = 0;

		modelViewMatrix.m02 = (float)extrinsicRotation[2];
		modelViewMatrix.m12 = (float)extrinsicRotation[5];
		modelViewMatrix.m22 = (float)extrinsicRotation[8];
		modelViewMatrix.m32 = 0;

		modelViewMatrix.m03 = (float)extrinsicTranslat[0];
		modelViewMatrix.m13 = (float)extrinsicTranslat[1];
		modelViewMatrix.m23 = (float)extrinsicTranslat[2];
		modelViewMatrix.m33 = 1;
		modelViewMatrix = modelViewMatrix.inverse;


		// Update unity camera transform
		Camera.main.transform.position = GetPositionFromModelView(modelViewMatrix);
		Camera.main.transform.rotation = GetRotationFromModelView(modelViewMatrix);
	}


	private Quaternion GetRotationFromModelView(Matrix4x4 modelViewMatrix)
	{
		Vector3 forward;
		forward.x = modelViewMatrix.m02;
		forward.y = -modelViewMatrix.m12;
		forward.z = modelViewMatrix.m22;

		Vector3 upwards;
		upwards.x = modelViewMatrix.m01;
		upwards.y = -modelViewMatrix.m11;
		upwards.z = modelViewMatrix.m21;

		return Quaternion.LookRotation(forward, upwards);
	}

	private Vector3 GetPositionFromModelView(Matrix4x4 modelViewMatrix)
	{
		Vector3 position;
		position.x = modelViewMatrix.m03;
		position.y = -modelViewMatrix.m13;
		position.z = modelViewMatrix.m23;
		return position;
	}
}