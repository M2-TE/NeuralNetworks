using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

namespace DeepQLearning
{
	public class DeepQLearning : MonoBehaviour
	{
		[SerializeField] private Vector2 center;
		[SerializeField] private float speed;
		[SerializeField] private Rigidbody2D rgb;

		private IEnumerator Start()
		{
			var settings = new DQNInitSettings()
			{
				
			};

			double[] stateArr = new double[] { center.x - transform.position.x, center.y - transform.position.y };
			var agent = new DQNAgent(settings, stateArr.Length, 4);

			while (true)
			{
				int action = agent.Act(stateArr);
				Vector2 movementVec = default;
				switch (action)
				{
					case 0:
						movementVec = new Vector2(1f, 0f) * Time.deltaTime * speed;
						break;

					case 1:
						movementVec = new Vector2(-1f, 0f) * Time.deltaTime * speed;
						break;

					case 2:
						movementVec = new Vector2(0f, 1f) * Time.deltaTime * speed;
						break;

					case 3:
						movementVec = new Vector2(0f, -1f) * Time.deltaTime * speed;
						break;
				}

				Vector2 pos = new Vector2(transform.position.x, transform.position.y);
				if((center - pos).sqrMagnitude > (center - pos + movementVec).sqrMagnitude)
				{
					Debug.Log("negative");
					agent.Learn(-1.0);
				}
				else
				{
					Debug.Log("positive");
					agent.Learn(1.0);
				}
				rgb.velocity = movementVec;
				yield return new WaitForFixedUpdate();
			}
		}
	}
	public delegate void BackpropAction();

	public static class Utils
	{
		private static bool return_v = false;
		private static double v_val = 0.0;
		public static double GetRandomGauss() // this can be improved (?)
		{
			if (return_v)
			{
				return_v = false;
				return v_val;
			}
			var u = 2 * Random.value - 1;
			var v = 2 * Random.value - 1;
			var r = u * u + v * v;
			if (r == 0 || r > 1) return GetRandomGauss();

			var c = Mathf.Sqrt(-2 * Mathf.Log(r) / r);

			v_val = v * c;
			return_v = true;

			return u * c;
		}

		public static double GetRandomGaussAlt() // do performance tests here
		{
			return default;
		}

		public static void FillMatrixWithRandomGaussianNumbers(Matrix m, double mu, double std)
		{
			for (int i = 0, n = m.w.Length; i < n; i++)
			{
				m.w[i] = mu + GetRandomGauss() * std;
			}
		}

		public static int ArgMax(double[] w)
		{
			double maxv = w[0];
			int maxix = 0;
			for (int i = 1, n = w.Length; i < n; i++)
			{
				double v = w[i];
				if (v > maxv)
				{
					maxix = i;
					maxv = v;
				}
			}
			return maxix;
		}

		public static void UpdateNet(Network net, double alpha) // IMPROVE HERE
		{
			if (net.W1 != null) UpdateMatrix(net.W1, alpha);
			if (net.b1 != null) UpdateMatrix(net.b1, alpha);
			if (net.W2 != null) UpdateMatrix(net.W2, alpha);
			if (net.b2 != null) UpdateMatrix(net.b2, alpha);
			//foreach (var mMatrix in m.Matrices)
			//{
			//	if (mMatrix != null)
			//	{
			//		UpdateMatrix(mMatrix, alpha);
			//	}
			//}
		}

		public static void UpdateMatrix(Matrix m, double alpha)
		{
			var n = m.rows * m.columns;
			for (var i = 0; i < n; i++)
			{
				if (m.dw[i] > 0)
				{
					m.w[i] += -alpha * m.dw[i];
					m.dw[i] = 0;
				}
			}
		}
	}

	public class Matrix
	{
		public readonly int rows;
		public readonly int columns;
		public readonly double[] w;
		public readonly double[] dw;

		public Matrix(int rows, int columns)
		{
			this.rows = rows;
			this.columns = columns;
			w = new double[rows * columns];
			dw = new double[rows * columns];
		}
		public Matrix(int rows, int columns, double mu, double std) : this(rows, columns)
		{
			Utils.FillMatrixWithRandomGaussianNumbers(this, mu, std);
		}

		public void setFrom(double[] arr)
		{
			for (int i = 0, n = arr.Length; i < n; i++)
			{
				w[i] = arr[i];
			}
		}
	}

	public class Network
	{
		public Matrix W1;
		public Matrix b1;
		public Matrix W2;
		public Matrix b2;
	}

	public class Graph
	{
		public readonly List<BackpropAction> backprop; // maybe use queue instead?
		public bool needs_backprop;

		public Graph(bool needs_backprop)
		{
			this.needs_backprop = needs_backprop;
			backprop = new List<BackpropAction>();
		}

		public void backward()
		{
			for (int i = 0, n = backprop.Count; i < n; i++)
			{
				backprop[i]();
			}
			//for (var i = backprop.Count - 1; i >= 0; i--)
			//{
			//	backprop[i](); // tick!
			//}

			// clear list?
		}

		public Matrix tanh(Matrix m)
		{
			var outMat = new Matrix(m.rows, m.columns);
			int n = m.w.Length;
			for (int i = 0; i < n; i++)
			{
				outMat.w[i] = Math.Tanh(m.w[i]);
			}

			if (needs_backprop)
			{
				void backward()
				{
					for (int i = 0; i < n; i++)
					{
						// grad for z = tanh(x) is (1 - z^2)
						double mwi = outMat.w[i];
						m.dw[i] += (1.0 - mwi * mwi) * outMat.dw[i];
					}
				}
				backprop.Add(backward);
			}

			return outMat;
		}

		public Matrix mul(Matrix m1, Matrix m2)
		{
			//if (m1.columns != m2.rows) Debug.LogException(new Exception("matmul dimensions misaligned"));

			var n = m1.rows;
			var d = m2.columns;
			Matrix outMat = new Matrix(n, d);

			for (int i = 0; i < m1.rows; i++)
			{
				for (int j = 0; j < m2.columns; j++)
				{
					double dot = 0.0;
					for (int k = 0; k < m2.columns; k++)
					{
						dot += m1.w[m1.columns * i + k] * m2.w[m2.columns * k + j];
					}
					outMat.w[d * i + j] = dot;
				}
			}

			if (needs_backprop)
			{
				void backward()
				{
					for (int i = 0; i < m1.rows; i++)
					{ // loop over rows of m1
						for (int j = 0; j < m2.columns; j++)
						{ // loop over cols of m2
							for (int k = 0; k < m1.columns; k++)
							{ // dot product loop
								var b = outMat.dw[d * i + j];
								m1.dw[m1.columns * i + k] += m2.w[m2.columns * k + j] * b;
								m2.dw[m2.columns * k + j] += m1.w[m1.columns * i + k] * b;
							}
						}
					}
				}

				backprop.Add(backward);
			}

			return outMat;
		}

		public Matrix add(Matrix m1, Matrix m2)
		{
			if (m1.w.Length != m2.w.Length) Debug.LogException(new Exception("matrix w lenghts dont match"));

			var outMat = new Matrix(m1.rows, m1.columns);
			for (int i = 0, n = m1.w.Length; i < n; i++)
			{
				outMat.w[i] = m1.w[i] + m2.w[i];
			}

			if (needs_backprop)
			{
				void backward()
				{
					for (int i = 0, n = m1.w.Length; i < n; i++)
					{
						m1.dw[i] += outMat.dw[i];
						m2.dw[i] += outMat.dw[i];
					}
				}
				backprop.Add(backward);
			}

			return outMat;
		}
	}

	public class Experience
	{
		public readonly Matrix s0;
		public readonly int a0;
		public readonly double r0;
		public readonly Matrix s1;
		public readonly int a1;

		public Experience(Matrix s0, int a0, double r0, Matrix s1, int a1)
		{
			this.s0 = s0;
			this.a0 = a0;
			this.r0 = r0;

			this.s1 = s1;
			this.a1 = a1;
		}
	}

	public class DQNInitSettings
	{
		public double gamma = 0.75;
		public double epsilon = 0.1;
		public double alpha = 0.01;

		public int experience_add_every = 25;
		public int experience_size = 5000;
		public int learning_steps_per_iteration = 10;
		public double tderror_clamp = 1.0;

		public int num_hidden_units = 100;
	}

	public class DQNAgent
	{
		#region Variables
		private Experience[] exp;
		private Network net;
		private Graph lastG;

		private int expi;
		private int t;
		private double tderror;

		private double r0;
		private Matrix s0, s1;
		private int a0, a1;

		private readonly int
			numStates,
			numActions;

		private readonly double
			gamma,
			epsilon,
			alpha;

		private readonly int
			experience_add_every,
			experience_size,
			learning_steps_per_iteration;
		private readonly double tderror_clamp;

		private readonly int num_hidden_units;
		#endregion

		public DQNAgent(DQNInitSettings opt, int numStates, int numActions)
		{
			this.numStates = numStates;
			this.numActions = numActions;

			gamma = opt.gamma;
			epsilon = opt.epsilon;
			alpha = opt.alpha;

			experience_add_every = opt.experience_add_every;
			experience_size = opt.experience_size;
			learning_steps_per_iteration = opt.learning_steps_per_iteration;
			tderror_clamp = opt.tderror_clamp;

			num_hidden_units = opt.num_hidden_units;

			Reset();
		}

		public void Reset()
		{
			net = new Network()
			{
				W1 = new Matrix(num_hidden_units, numStates, 0.0, 0.01),
				b1 = new Matrix(num_hidden_units, 1, 0.0, 0.01),
				W2 = new Matrix(numActions, num_hidden_units, 0.0, 0.01),
				b2 = new Matrix(numActions, 1, 0.0, 0.01)
			};

			exp = new Experience[experience_size];
			expi = 0;

			t = 0;

			r0 = default;
			s0 = default;
			s1 = default;
			a0 = default;
			a1 = default;

			tderror = 0;
		}

		public Matrix ForwardQ(Network net, Matrix s, bool needs_backprop)
		{
			var G = new Graph(needs_backprop);
			var a1mat = G.add(G.mul(net.W1, s), net.b1);
			var h1mat = G.tanh(a1mat);
			var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);

			lastG = G; // is this really necessary?
			return a2mat;
		}

		public int Act(double[] slist)
		{
			// convert to a Mat column vector
			Matrix s = new Matrix(numStates, 1);
			s.setFrom(slist);

			// epsilon greedy policy
			int a = 0;
			if (Random.value < epsilon)
			{
				// return random action
				a = Random.Range(0, numActions);
			}
			else
			{
				// greedy wrt Q function
				var amat = ForwardQ(net, s, false);
				a = Utils.ArgMax(amat.w);
			}

			// shift state memory
			s0 = s1;
			a0 = a1;
			s1 = s;
			a1 = a;

			return a;
		}

		public void Learn(double r1)
		{
			// perform an update on Q function
			if (r0 != 0.0 && alpha > 0)
			{
				// learn from this tuple to get a sense of how "surprising" it is to the agent
				double tderror = LearnFromTuple(s0, a0, r0, s1, a1);
				this.tderror = tderror; // a measure of surprise

				// decide if we should keep this experience in the replay
				if (t % experience_add_every == 0)
				{
					var singleExperience = new Experience(s0, a0, r0, s1, a1);
					exp[expi] = singleExperience;

					expi++; // can be improved
					if (expi > experience_size - 1)
					{
						expi = 0;
					}
				}
				t += 1;

				// sample some additional experience from replay memory and learn from it
				for (int k = 0; k < learning_steps_per_iteration; k++)
				{
					//var ri = randi(0, exp.length); // todo: priority sweeps?
					var ri = Random.Range(0, expi); // changed this
					var e = exp[ri];
					LearnFromTuple(e.s0, e.a0, e.r0, e.s1, e.a1);
				}
			}
			r0 = r1;
		}

		private double LearnFromTuple(Matrix s0, int a0, double r0, Matrix s1, int a1)
		{
			// want: Q(s,a) = r + gamma * max_a' Q(s',a')

			// compute the target Q value
			Matrix tmat = ForwardQ(net, s1, false);
			double qmax = r0 + gamma * tmat.w[Utils.ArgMax(tmat.w)];

			// now predict
			Matrix pred = ForwardQ(net, s0, true);

			double tderror = pred.w[a0] - qmax;
			double clamp = tderror_clamp; // obsolete
			if (Math.Abs(tderror) > clamp) // huber loss to robustify || THIS CAN BE IMPROVED
			{
				if (tderror > clamp) tderror = clamp;
				if (tderror < -clamp) tderror = -clamp;
			}

			pred.dw[a0] = tderror; // pred is lost after func call, so why assign here?
			lastG.backward(); // compute gradients on net params

			// update net
			Utils.UpdateNet(net, alpha);
			return tderror;
		}

		public void Save()
		{

		}

		public void Load()
		{

		}
	}
}
