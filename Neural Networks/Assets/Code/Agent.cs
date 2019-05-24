using NeuralNetworks;
using UnityEngine;

public class Agent : MonoBehaviour
{
	public NeuralNetworkCompact NN;
	public Camera cam;
	[SerializeField] private float speed;

	private void Update()
	{

		Vector3 mousePos = cam.ScreenToWorldPoint(Input.mousePosition, Camera.MonoOrStereoscopicEye.Mono);
		mousePos.z = 0f;
		float rot = Vector3.Angle(-transform.up, mousePos - transform.position);

		var output = NN.FeedForward(new float[] { rot });

		transform.Rotate(new Vector3(0f, 0f, output[0]));
		transform.Translate(-transform.up * Time.deltaTime * speed);
		Debug.DrawRay(transform.position, -transform.up, Color.red);
	}
}
