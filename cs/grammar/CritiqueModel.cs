using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Grammar
{
    abstract public class CritiqueModel : Grammar.TFModel
    {
        [Serializable]
        public class Critique
        {
            [Serializable]
            public struct CritiqueTarget
            {
                public string target;
                public float prob;
                public CritiqueTarget(string target_value, float prob_value)
                {
                    target = target_value;
                    prob = prob_value;
                }
            }
            public int critique_start;
            public int critique_len;
            public string source_string;
            public List<CritiqueTarget> targets;
            public string model;
            public string critique_description;
            public string critique_name;
        };
        [Serializable]
        public class DeterminerCritique : Critique
        {
            /// <summary>
            /// Length of the following noun phrase in number of tokens
            /// </summary>
            public int noun_phrase_length;
            /// <summary>
            /// Start token index
            /// </summary>
            public int start_token;
        }
        abstract public List<CritiqueModel.Critique> Eval(string sentence);
        
    }
}
