using System.Collections;
using System.Linq;
using RainForce.Agents;
using RainForce.Models;
using UnityEngine;
using Random = System.Random;

namespace Example
{
    class ExampleDQN : MonoBehaviour
    {
		private IEnumerator Start()
		{
			var rnd = new Random();
			int max = 10;
			int min = 1;
			int nextPrint = 0, act1 = 0, act0 = 0;
			double total = 0, correct = 0;
			var state = new[] { rnd.Next(min, max), rnd.Next(min, max), rnd.Next(min, max), rnd.Next(min, max) };
			var opt = new TrainingOptions
			{
				Alpha = 0.001,
				Epsilon = 0,
				ErrorClamp = 0.002,
				ExperienceAddEvery = 10,
				ExperienceSize = 1000,
				ExperienceStart = 0,
				HiddenUnits = 5,
				LearningSteps = 400
			};
			//we take 4 states i.e random numbers between 1 and 10
			//we have 2 actions 1 if average of set is >5 and 0 if otherwise
			//we reward agent with 1 for every correct and -1 otherwise
			var agent = new DQNAgent(opt, state.Length, 2);


			while (total < 50000)
			{
				state = new[] { rnd.Next(min, max), rnd.Next(min, max), rnd.Next(min, max), rnd.Next(min, max) };
				var action = agent.Act(state);
				if (action == 1)
				{
					act1++;
				}
				else
				{
					act0++;
				}

				if (state.Average() > 5 && action == 1)
				{
					agent.Learn(1);
					correct++;
				}
				else if (state.Average() <= 5 && action == 0)
				{
					agent.Learn(1);
					correct++;
				}
				else
				{
					agent.Learn(-1);
				}

				total++;
				//nextPrint++;
				if (total >= nextPrint)
				{
					Debug.Log("Score: " + (correct / total).ToString("P") + "Epoch: " + nextPrint);
					Debug.Log("Action 1: " + act1 + " Action 0: " + act0);
					//correct = total = act0 = act1 = 0; //reset
					nextPrint += 1000;
				}
				yield return null;
			}
			Debug.Log("Score: " + (correct / total).ToString("P"));
			Debug.Log("End");
			//File.AppendAllText(AppDomain.CurrentDomain.BaseDirectory + "DNQ.trr", agent.AgentToJson());
		}
    }

    public class MyDPAgent : DPAgent
    {
        protected override int[] GetAllowedActions(int s)
        {
            return new[] {0, 1, 2, 3};
        }

        protected override int Reward(int s, int a, int d)
        {
            return s * a * d;
        }

        protected override int NextStateDistribution(int s, int a)
        {
            return s * a;
        }
    }
}
