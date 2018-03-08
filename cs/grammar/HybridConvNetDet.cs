using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using TensorFlow;
using Tokenex;

namespace Grammar
{
    public class HybridConvNetDet : CritiqueModel
    {
        Dictionary<string, int> _vocab;
        Tokenizer _tokenizer;
        string _pad_token = "<pad>";
        string _start_token = "<s>";
        int _pad_idx;
        int _start_idx;
        int _num_words_before;
        int _num_words_after;

        bool _all_lowercase = false;

        int _word_len = -1;
        int _ccnn_num_words = -1;
        bool _ccnn_skip_nonalphanumeric = false;
        bool _enable_noun_phrase_search = false;
        bool _insert_only = false;
        string[] _keywords;

        /// <summary>
        /// Construct HybridConvNetDet including loading the model.
        /// <param name="modelDirectory">The directory from which to load the model<param>
        /// <param name="cacheDirectoryOrNull">If given, the location of a local directory where newer model files will be copied before they are loaded</param>
        /// </summary>
        public HybridConvNetDet(bool insert_only, bool enable_noun_phrase_search, string modelDirectory, string cacheDirectoryOrNull = null)
        {
            string actualDirectory = ActualDirectory(modelDirectory, cacheDirectoryOrNull);
            bool result = this.Load(actualDirectory);
            Trace.Assert(result, string.Format("Error loading directory '{0}'", actualDirectory));
            _enable_noun_phrase_search = enable_noun_phrase_search;
            _insert_only = insert_only;
        }

        private string ActualDirectory(string modelDirectory, string cacheDirectoryOrNull)
        {
            if (cacheDirectoryOrNull == null)
            {
                return modelDirectory;
            }
            string cacheDirectory = Path.Combine(cacheDirectoryOrNull, modelDirectory.Substring(Path.GetPathRoot(modelDirectory).Length));
            (new DirectoryInfo(cacheDirectory)).Create(); //Create directory if doesn't exist
            foreach (string fromFile in Directory.EnumerateFiles(modelDirectory))
            {
                string toFile = Path.Combine(cacheDirectory, Path.GetFileName(fromFile));
                if ((new FileInfo(fromFile)).LastWriteTimeUtc > (new FileInfo(toFile).LastWriteTimeUtc)) //Surprisingly, this works even if toFile doesn't exist
                {
                    Debug.WriteLine("Copying from '{0}' to '{1}'", fromFile, toFile);
                    File.Delete(toFile); //Use a temp file to be sure that incomplete copies don't stop later copies. This operation works even if the toFile doesn't exist
                    File.Delete(toFile + ".tmp");
                    File.Copy(fromFile, toFile + ".tmp");
                    File.Move(toFile + ".tmp", toFile);
                }
            }
            return cacheDirectory;
        }


        new protected bool Load(string model_dir)
        { 
            bool result = base.Load(model_dir); ;
            // TODO: load the appropriate tokenizer based on param
            if (result)
            {
                _tokenizer = new Tokenizer();
                _vocab = Grammar.Utils.LoadVocab(Path.Combine(model_dir, "vocab.json"));
                if (_params.ContainsKey("pad_token"))
                    _pad_token = _params["pad_token"].ToString();
                _pad_idx = _vocab[_pad_token];
                if (_params.ContainsKey("start_token"))
                    _start_token = _params["start_token"].ToString();
                _start_idx = _vocab[_start_token];

                _num_words_before = Convert.ToInt32(_params["num_words_before"]);
                _num_words_after = Convert.ToInt32(_params["num_words_after"]);

                _word_len = Convert.ToInt32(_params["word_len"]);
                _ccnn_num_words = Convert.ToInt32(_params["ccnn_num_words"]);

                if (_params.ContainsKey("ccnn_skip_nonalphanumeric"))
                    _ccnn_skip_nonalphanumeric = Convert.ToBoolean(_params["ccnn_skip_nonalphanumeric"]);
                _keywords = new string[((Newtonsoft.Json.Linq.JArray)(_params["keywords"])).Count + 1];
                _keywords[0] = "";
                for (int i = 1; i < _keywords.Length; i++)
                    _keywords[i] = (string)(((Newtonsoft.Json.Linq.JArray)(_params["keywords"]))[i - 1]);

                if (_params.ContainsKey("all_lowercase"))
                    _all_lowercase = Convert.ToBoolean(_params["all_lowercase"]);
            }
            return result;
        }

        bool HasAlphaNumeric(string str)
        {
            return Regex.Matches(str, @"[a-zA-Z0-9]").Count>0; 
        }
        public override List<Critique> Eval(string sentence)
        {
            if (_all_lowercase)
                sentence = sentence.ToLower();

            var extokens = _tokenizer.TokenizeEx(sentence);

            var tokensNoDet = new List<Tokenizer.Token>();
            Dictionary<int,int> detPosition = new Dictionary<int, int>();
            Dictionary<int, int> inputDet = new Dictionary<int, int>();
            int new_token_idx = 0;
            int old_token_idx = 0;
            List<int> original = new List<int>();
            // remove all the determiners
            foreach (var token in extokens)
            {
                int det = Array.IndexOf(_keywords, token.token);
                //if (false)// (det >= 0)
                if ((!_insert_only) && (det >= 0))
                {
                    if (inputDet.ContainsKey(new_token_idx))
                    {
                        inputDet[new_token_idx] = -1;
                    }
                    else
                    {
                        inputDet[new_token_idx] = det;
                    }
                }
                else
                {
                    if (!inputDet.ContainsKey(new_token_idx))
                    {
                        inputDet[new_token_idx] = 0;
                    }
                    tokensNoDet.Add(token);
                    detPosition[new_token_idx] = old_token_idx;
                    new_token_idx += 1;
                }
                old_token_idx += 1;
            }

            var critiques = EvalSimple(tokensNoDet, inputDet);

            List<Critique> result = new List<Critique>();

            foreach (var critique in critiques) { 
                int start_tok_idx = 0;
                if (critique.start_token > 0)
                    start_tok_idx = detPosition[critique.start_token - 1] + 1;

                Critique newcritique = new Critique();

                newcritique.critique_start = extokens.ElementAt(start_tok_idx).start;
                if (_enable_noun_phrase_search && (critique.noun_phrase_length > 0))
                {
                    int end_tok_idx = detPosition[critique.start_token + critique.noun_phrase_length];
                    newcritique.critique_len = extokens.ElementAt(end_tok_idx - 1).start + extokens.ElementAt(end_tok_idx - 1).len - newcritique.critique_start;
                    newcritique.source_string = sentence.Substring(newcritique.critique_start, newcritique.critique_len);
                    //                    newcritique.targets = critique.targets;
                    newcritique.targets = new List<Critique.CritiqueTarget>();
                    foreach (var target in critique.targets)
                    {
                        Critique.CritiqueTarget new_target = new Critique.CritiqueTarget();
                        new_target.prob = target.prob;
                        new_target.target = target.target + " " + sentence.Substring(extokens.ElementAt(start_tok_idx).start,
                            extokens.ElementAt(end_tok_idx-1).start + extokens.ElementAt(end_tok_idx - 1).len - newcritique.critique_start);
                        newcritique.targets.Add(new_target);
                    }
                }
                else
                {
                    int end_tok_idx = detPosition[critique.start_token];
                    newcritique.critique_len = extokens.ElementAt(end_tok_idx).start - newcritique.critique_start;
                    newcritique.source_string = sentence.Substring(newcritique.critique_start, newcritique.critique_len);
                    newcritique.targets = new List<Critique.CritiqueTarget>();
                    foreach (var target in critique.targets)
                    {
                        Critique.CritiqueTarget new_target = new Critique.CritiqueTarget();
                        new_target.prob = target.prob;
                        if (target.target.Length > 0)
                            new_target.target = target.target + " ";
                        else
                            new_target.target = target.target;
                        newcritique.targets.Add(new_target);
                    }
                }
                result.Add(newcritique);
            }
            return result;
        }
        int argmax1(float[,] matrix, int j)
        {
            // argmax over dimension #1 (count starting from 0)
            int result = 0;
            float max_value = 0.0f;
            for (var i = 0; i < matrix.GetLength(1); i++)
            {
                if (matrix[j, i] > max_value)
                {
                    max_value = matrix[j, i];
                    result = i;
                }
            }
            return result;
        }
        public int GetNounPhraseLength(int[] tokens_idx, List<Tokenizer.Token> extokens, int location, int max_len)
        {
            max_len = Math.Min(extokens.Count - location, max_len);
            int[][] inputs = new int[max_len][];
            string[][] inputs_dbg = new string[max_len][];
            int[][] char_cnn_path = new int[max_len][];
            // generate inputs
            for (var i = 0; i < max_len; i++)
            {
                char_cnn_path[i] = new int[_word_len];
                //                int k = 0;
                int words_seen = 0;
                string char_cnn_str = "";
                for (var j = 0; j < tokens_idx.Length - i - location; j++)
                {
                    string word = extokens[j + i + location].token;
                    if ((!_ccnn_skip_nonalphanumeric) || (HasAlphaNumeric(word)))
                    {
                        words_seen += 1;
                        if (char_cnn_str.Length > 0)
                            char_cnn_str += " ";
                        char_cnn_str += word;
                    }
                    if (words_seen >= _ccnn_num_words)
                        break;
                }
                for (var j = 0; j < Math.Min(_word_len, char_cnn_str.Length); j++)
                    char_cnn_path[i][j] = char_cnn_str[j];
                inputs[i] = new int[_num_words_before + _num_words_after];
                inputs_dbg[i] = new string[_num_words_before + _num_words_after];
                for (int j = 0; j < _num_words_after + _num_words_before; j++)
                {
                    if (location + j - _num_words_before < -1)
                    {
                        inputs[i][j] = _pad_idx;
                        inputs_dbg[i][j] = _pad_token;
                    }
                    else if (location + j - _num_words_before == -1)
                    {
                        inputs[i][j] = _start_idx;
                        inputs_dbg[i][j] = _start_token;
                    }
                    else if ((j< _num_words_after) && (location + j - _num_words_before < tokens_idx.Length))
                    {
                        inputs[i][j] = tokens_idx[location + j - _num_words_before];
                        inputs_dbg[i][j] = extokens[location + j - _num_words_before].token;
                    }
                    else if ((j >= _num_words_after) && (i + location + 1 + j - _num_words_before < tokens_idx.Length))
                    {
                        inputs[i][j] = tokens_idx[location + i + j - _num_words_before];
                        inputs_dbg[i][j] = extokens[location + i + 1 + j - _num_words_before].token;
                    }
                    else
                    {
                        inputs[i][j] = _pad_idx;
                        inputs_dbg[i][j] = _pad_token;
                    }
                }
            }
            TFTensor tfsentence = new TFTensor(inputs);
            TFTensor tfword = new TFTensor(char_cnn_path);
            Dictionary<string, TFTensor> tfinputs = new Dictionary<string, TFTensor>();
            string[] outputs = { "sm_decision" };
            tfinputs["sentence"] = tfsentence;
            tfinputs["word"] = tfword;
            var tfoutputs = Eval(tfinputs, outputs);
            TFTensor sm_decision = tfoutputs["sm_decision"];
            object o = sm_decision.GetValue();
            float[,] probs = (float[,])o;

            int result = 0;
            while ((result < max_len) && (argmax1(probs, result) >0)) result++;

            return result;
        }
        public static int[] IndexTokenListToArray(List<Tokenizer.Token> tokens, Dictionary<string, int> vocab, string unk_token = "unk")
        {
            int[] token_idx = new int[tokens.Count];
            int i = 0;
            foreach (var token in tokens)
            {
                if (vocab.ContainsKey(token.token))
                    token_idx[i] = (vocab[token.token]);
                else
                    token_idx[i] = (vocab[unk_token]);
                i++;
            }
            return token_idx;
        }

        public List<DeterminerCritique> EvalSimple(List<Tokenizer.Token> extokens, Dictionary<int, int> original)
        {
            List<DeterminerCritique> result = new List<DeterminerCritique>();
            int[] tokens_idx = IndexTokenListToArray(extokens, _vocab);
            float[,] probs = EvalRaw(tokens_idx, extokens);
            for (int token_idx = 0; token_idx < extokens.Count; token_idx++)
            {
                // what do we recommend?
                int top1 = -1, top2 = -1;
                float top1v = 0, top2v = 0;
                for (int choice_idx = 0; choice_idx < probs.GetLength(1); choice_idx++)
                {
                    if (probs[token_idx,choice_idx] > top1v)
                    {
                        if (top1v > top2v)
                        {
                            top2v = top1v;
                            top2 = top1;
                        }
                        top1v = probs[token_idx, choice_idx];
                        top1 = choice_idx;
                    }
                    else if (probs[token_idx, choice_idx] > top2v)
                    {
                        top2v = probs[token_idx, choice_idx];
                        top2 = choice_idx;
                    }
                }
                if (
                    ((top1 != original[token_idx]) && (top2 != original[token_idx])) 
                    )
                {
                    // critique
                    int original_idx = original[token_idx];
                    float original_prob;
                    if (original_idx >= 0)
                        original_prob = probs[token_idx, original_idx];
                    else
                        original_prob = 0;

                    DeterminerCritique c = new DeterminerCritique();
                    c.targets = new List<Critique.CritiqueTarget>();
                    c.targets.Add(new Critique.CritiqueTarget(_keywords[top1], 
                        (probs[token_idx, top1])/(probs[token_idx, top1] + original_prob)
                        ));
                    if (top1 > 0)
                    {
                        c.targets.Add(new Critique.CritiqueTarget(_keywords[top2],
                            (probs[token_idx, top2]) / (probs[token_idx, top2] + original_prob)
                            ));
                    }
                    c.start_token = token_idx;
                    if (top1 > 0)
                        c.noun_phrase_length = GetNounPhraseLength(tokens_idx, extokens, token_idx, 5);
                    else
                        c.noun_phrase_length = 0;
                    result.Add(c);

//                    System.Console.WriteLine("{3}: {0} -> {1},{2}", original[token_idx], top1, top2, token_idx);
                }
            }
            return result;
        }

        public float[,] EvalRaw(int[] tokens_idx, List<Tokenizer.Token> extokens)
        {
            /*
            string[] tokens = new string[extokens.Count];
            int tokeni = 0;
            foreach (var extoken in extokens)
                tokens[tokeni++] = extoken.token;
            int[] tokens_idx = Grammar.Utils.IndexText(tokens, _vocab);
            */
            int[][] inputs = new int[tokens_idx.Length][];
            string[][] inputs_dbg = new string[tokens_idx.Length][];
            int[][] char_cnn_path = new int[tokens_idx.Length][];
            // generate inputs
            for (var i = 0; i < tokens_idx.Length; i++)
            {
                char_cnn_path[i] = new int[_word_len];
//                int k = 0;
                int words_seen = 0;
                string char_cnn_str = "";
                for (var j = 0; j < tokens_idx.Length-i; j++)
                {
                    string word = extokens[j + i].token;
                    if ((!_ccnn_skip_nonalphanumeric) || (HasAlphaNumeric(word)))
                    {
                        words_seen += 1;
                        if (char_cnn_str.Length > 0)
                            char_cnn_str += " ";
                        char_cnn_str += word;
                    }
                    if (words_seen >= _ccnn_num_words)
                        break;
                }
                for (var j = 0; j < Math.Min(_word_len, char_cnn_str.Length); j++)
                    char_cnn_path[i][j] = char_cnn_str[j];
                inputs[i] = new int[_num_words_before + _num_words_after];
                inputs_dbg[i] = new string[_num_words_before + _num_words_after];
                for (int j = 0; j < _num_words_after + _num_words_before; j++)
                {
                    if (i + j - _num_words_before < -1)
                    {
                        inputs[i][j] = _pad_idx;
                        inputs_dbg[i][j] = _pad_token;
                    }
                    else if (i+j - _num_words_before == -1)
                    {
                        inputs[i][j] = _start_idx;
                        inputs_dbg[i][j] = _start_token;
                    }
                    else if (i + j - _num_words_before < tokens_idx.Length)
                    {
                        inputs[i][j] = tokens_idx[i + j - _num_words_before];
                        inputs_dbg[i][j] = extokens[i + j - _num_words_before].token;
                    }
                    else {
                        inputs[i][j] = _pad_idx;
                        inputs_dbg[i][j] = _pad_token;
                    }
                }
            }
            TFTensor tfsentence = new TFTensor(inputs);
            TFTensor tfword = new TFTensor(char_cnn_path);
            Dictionary<string, TFTensor> tfinputs = new Dictionary<string, TFTensor>();
            string[] outputs = { "sm_decision" };
            tfinputs["sentence"] = tfsentence;
            tfinputs["word"] = tfword;
            var tfoutputs = Eval(tfinputs, outputs);
            TFTensor sm_decision = tfoutputs["sm_decision"];
            object o = sm_decision.GetValue();
            float[,] probs = (float[,])o;

            /*
            for (int pos = 0; pos < probs.GetLength(0); pos++)
            {
                int max_idx = 0;
                int max_prob = 0;
//                for (int sm_idx = 0; sm_idx < probs.GetLength(1); sm_idx++)
                    //System.Console.WriteLine("{0} {1} {2:0.00}", pos, sm_idx, probs[pos, sm_idx]);
            }
            */
            return probs;
        }
    }
}
