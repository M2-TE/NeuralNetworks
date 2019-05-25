using NeuralNetworks;
using UnityEngine;

public class Agent : MonoBehaviour
{
	public NeuralNetwork NN;
	public Camera cam;
	[SerializeField] private float speed;
	[SerializeField] private float turnSpeed;

	private void Update()
	{
		Vector3 mousePos = cam.ScreenToWorldPoint(Input.mousePosition, Camera.MonoOrStereoscopicEye.Mono);
		mousePos.z = 0f;

		float angle = Vector3.Angle(transform.up, mousePos - transform.position);
		var output = NN.RequestDecision(new float[] { angle });
		transform.Rotate(0f, 0f, output[0] * turnSpeed);
		transform.position = transform.position + transform.up * speed * Time.deltaTime;

		//NN.Fitness += 1f / angle;
		if((transform.position - mousePos).sqrMagnitude < .1f)
		{
			NN.Fitness += 5f;
			// trigger next gen
		}
		else if (angle < 1f || (transform.position - mousePos).sqrMagnitude < 1f)
		{
			NN.Fitness += 1f;
		}
		else
		{
			NN.Fitness -= .1f;
			NN.Mutate();
		}
	}
}
