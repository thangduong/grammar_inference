using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.IO;

namespace Grammar
{
    public class Utils
    {
        public static Dictionary<string, int> LoadVocab(string vocab_file)
        {
            string vocab_json = File.ReadAllText(vocab_file);
            return JsonConvert.DeserializeObject<Dictionary<string, int>>(vocab_json);
        }
        public static int[] IndexText(string[] tokens, Dictionary<string, int> vocab, string unk_token = "unk")
        {
            int[] token_idx = new int[tokens.Length];
            for (var i = 0; i < tokens.Length; i++)
            {
                if (vocab.ContainsKey(tokens[i]))
                    token_idx[i]=(vocab[tokens[i]]);
                else
                    token_idx[i] = (vocab[unk_token]);
            }
            return token_idx;
        }
        public static List<int> IndexText(List<string> token_list, Dictionary<string, int> vocab, string unk_token = "unk")
        {
            List<int> result = new List<int>();
            foreach (var token in token_list)
            {
                if (vocab.ContainsKey(token))
                    result.Add(vocab[token]);
                else
                    result.Add(vocab[unk_token]);
            }
            return result;
        }
    }
}
