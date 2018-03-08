using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tokenex;
using TensorFlow;

namespace Grammar
{
    /// <summary>
    /// Wraps up the inference portion of a tensorflow generative LM
    /// </summary>
    public class GenerativeLM : TFModel
    {
        Dictionary<string, int> _vocab;
        Tokenizer _tokenizer;
        public float ScoreSentence(string sentence)
        {
            // no batching
            float result = 1.0f;
            int num_steps = (int)(_params["num_steps"]);// Int32.Parse(_params["num_steps"]);
            int num_layers = (int)(_params["num_layers"]); // Int32.Parse(_params["num_layers"]);
            int cell_size = (int)(_params["cell_size"]);// Int32.Parse(_params["cell_size"]);
            bool lowercase = (bool)(_params["all_lowercase"]);// Boolean.Parse(_params["all_lowercase"]);
            int multiplier = 2;
            int batch_size = 1;
            if (lowercase)
                sentence = "<s>" + sentence.ToLower();
            else
                sentence = "<s>" + sentence;
            var token_txt = _tokenizer.Tokenize(sentence);
//            var token_txt_str = String.Join(" ", token_txt.ToArray());
//            System.Console.WriteLine(token_txt_str);
            var tokens = Utils.IndexText(token_txt, _vocab).ToArray();
            float[] stateTensor = new float[multiplier * cell_size * num_layers * batch_size];
            TFShape stateShape = new TFShape(num_layers, multiplier, batch_size, cell_size);
            int[] sentenceTensor = new int[num_steps];
            TFShape sentenceShape = new TFShape(batch_size, num_steps);
            TFTensor state = TFTensor.FromBuffer(stateShape, stateTensor, 0, stateTensor.Length);

            Dictionary<string,TFTensor> inputs = new Dictionary<string, TFTensor>();
            string[] outputs = { "output_logits_sm" , "final_state" };
            inputs["state"] = state;

            int curIndex = 0;
            int offset = 0;
            while (curIndex < tokens.Length)
            {
                // this copy can be removed, but it's small enough that it's not an issue right now
                Array.Copy(tokens, curIndex, sentenceTensor, 0, Math.Min(num_steps, tokens.Length - curIndex));
                TFTensor x = TFTensor.FromBuffer(sentenceShape, sentenceTensor, 0, sentenceTensor.Length);
                inputs["x"] = x;
                Dictionary<string, TFTensor> results = Eval(inputs, outputs);
                inputs["state"] = results["final_state"];
                object output_logits_sm = results["output_logits_sm"].GetValue();
                // skip #0
                for (var i = 0; i < num_steps; i++)
                {                 
                    curIndex += 1;
                    if (curIndex == tokens.Length)
                        break;
                    float cur_prob = ((float[,,])output_logits_sm)[0, curIndex-1-offset, tokens[curIndex]];
//                    System.Console.WriteLine("{0} {1} {2}", token_txt.ElementAt(curIndex), tokens.ElementAt(curIndex), cur_prob);
                    result *= cur_prob;
                }
                offset += num_steps;
            }
            return result;
        }
        public Tuple<float, TFTensor> ScoreSentenceFast(string sentence)
        {
            // NOTE: this version doesn't work!  yet.
            // no batching
            float result = 1.0f;
            int num_steps = (int)(_params["num_steps"]);// Int32.Parse(_params["num_steps"]);
            int num_layers = (int)(_params["num_layers"]);// Int32.Parse(_params["num_layers"]);
            int cell_size = (int)(_params["cell_size"]);// Int32.Parse(_params["cell_size"]);
            bool lowercase = (bool)(_params["all_lowercase"]);// Boolean.Parse(_params["all_lowercase"]);
            int multiplier = 2;
            int batch_size = 1;
            if (lowercase)
                sentence = "<s>" + sentence.ToLower();
            else
                sentence = "<s>" + sentence;
            var tokens = Utils.IndexText(_tokenizer.Tokenize(sentence), _vocab).ToArray();
            float[] stateTensor = new float[multiplier * cell_size * num_layers * batch_size];
            TFShape stateShape = new TFShape(num_layers, multiplier, batch_size, cell_size);
            int[] sentenceTensor = new int[num_steps];
            TFShape sentenceShape = new TFShape(batch_size, num_steps);
            TFTensor state = TFTensor.FromBuffer(stateShape, stateTensor, 0, stateTensor.Length);
            int[] smeiTensor = new int[num_steps];
            TFShape smeiShape = new TFShape(batch_size, num_steps);

            Dictionary<string, TFTensor> inputs = new Dictionary<string, TFTensor>();
            string[] outputs = { "output_single_sm", "final_state" };
            inputs["state"] = state;

            int curIndex = 0;
            int offset = 0;
            while (curIndex < tokens.Length)
            {
                // NOTE: no need to copy.  Fix this!
                Array.Copy(tokens, curIndex + 1, smeiTensor, 0, Math.Min(num_steps, tokens.Length - curIndex - 1));
                TFTensor smei = TFTensor.FromBuffer(smeiShape, smeiTensor, 0, sentenceTensor.Length);
                Array.Copy(tokens, curIndex, sentenceTensor, 0, Math.Min(num_steps, tokens.Length - curIndex));
                TFTensor x = TFTensor.FromBuffer(sentenceShape, sentenceTensor, 0, sentenceTensor.Length);
                inputs["x"] = x;
                inputs["smei"] = smei;
                Dictionary<string, TFTensor> results = Eval(inputs, outputs);
                inputs["state"] = results["final_state"];
                object output_single_sm = results["output_single_sm"].GetValue();
                // skip #0
                for (var i = 0; i < num_steps; i++)
                {
                    curIndex += 1;
                    if (curIndex == tokens.Length)
                        break;
                    float cur_prob = ((float[,])output_single_sm)[0, i];
                    result *= cur_prob;
                }
                offset += num_steps;
            }
            return new Tuple<float, TFTensor>(result, inputs["state"]);
        }

        public float[] ScoreFIB(string prefix, string postfix, string[] choices)
        {
            int num_steps = (int)(_params["num_steps"]);// Int32.Parse(_params["num_steps"]);
            int num_layers = (int)(_params["num_layers"]);// Int32.Parse(_params["num_layers"]);
            int cell_size = (int)(_params["cell_size"]);// Int32.Parse(_params["cell_size"]);
            bool lowercase = (bool)(_params["all_lowercase"]);// Boolean.Parse(_params["all_lowercase"]);
            int multiplier = 2;
            int batch_size = choices.Length;
            // generate state with prefix!
            // score choices + postfix in a single batch
            float[] result = new float[choices.Length];

            if (lowercase)
                prefix = "<s>" + prefix.ToLower();
            else
                prefix = "<s>" + prefix;
            Tuple<float, TFTensor> first_part = ScoreSentenceFast(prefix);

            float[] stateTensor = new float[multiplier * cell_size * num_layers * batch_size];
            TFShape stateShape = new TFShape(num_layers, multiplier, batch_size, cell_size);
            int[] sentenceTensor = new int[num_steps];
            TFShape sentenceShape = new TFShape(batch_size, num_steps);
            TFTensor state = TFTensor.FromBuffer(stateShape, stateTensor, 0, stateTensor.Length);
            int[] smeiTensor = new int[num_steps];
            TFShape smeiShape = new TFShape(batch_size, num_steps);

            Dictionary<string, TFTensor> inputs = new Dictionary<string, TFTensor>();
            string[] outputs = { "output_single_sm", "final_state" };
            inputs["state"] = state;

            foreach (var choice in choices)
            {
                string sentence = choices + " " + postfix;
                var tokens = Utils.IndexText(_tokenizer.Tokenize(sentence), _vocab).ToArray();
            }

            return result;
        }

        public override bool Load(string model_dir)
        {
            // TODO: load the appropriate tokenizer based on param
            _tokenizer = new Tokenizer();
            _vocab = Grammar.Utils.LoadVocab(Path.Combine(model_dir, "vocab.json"));
            return base.Load(model_dir);
        }
    }
}
