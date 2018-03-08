using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using TensorFlow;
using System;

namespace Grammar
{
    public class TFModel
    {
        protected Dictionary<string, object> _params;
        TFGraph _graph;
        TFSession _session;
        string _prefix;

        public virtual object GetParameter(string param_name, object default_value)
        {
            if (_params.ContainsKey(param_name))
                return _params[param_name];
            else
                return default_value;
        }

        /// <summary>
        /// Load param and model from a model directory
        /// </summary>
        /// <param name="model_dir">directory containing param and model (graphdef) file</param>
        /// <returns>true if successful, false if failed</returns>
        public virtual bool Load(string model_dir)
        {
            bool result = false;

            try
            {
                string param_file = Path.Combine(model_dir, "params.json");
                // looks for params.json file, load model
                string params_json = File.ReadAllText(param_file);
                _params = JsonConvert.DeserializeObject<Dictionary<string, object>>(params_json);

                _prefix = (string)(_params["model_name"]);
                string graphdef_filepath = Path.Combine(model_dir, _prefix + ".graphdef");

                _graph = new TFGraph();
                TFImportGraphDefOptions opts = new TFImportGraphDefOptions();
                TFBuffer buff = new TFBuffer(File.ReadAllBytes(graphdef_filepath));
                opts.SetPrefix(_prefix);
                _graph.Import(buff, opts);
                _session = new TFSession(_graph);
                result = true;
            } catch (Exception)
            {
                // somethign went wrong
            }
     
            return result;
        }

        /// <summary>
        /// Evaluate the model given input and get outputs
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        /// <returns></returns>
        public Dictionary<string, TFTensor> Eval(Dictionary<string, TFTensor> inputs, string[] outputs)
        {
            Dictionary<string, TFTensor> result = new Dictionary<string, TFTensor>();
            var runner = _session.GetRunner();
            foreach (var input in inputs)
            {
                var input_node = _graph[_prefix + "/" + input.Key];
                runner.AddInput(input_node[0], input.Value);
            }

            // now ad "is_training" and make it false
            var is_training_node = _graph[_prefix + "/" + "is_training"];
            if (is_training_node != null)
                runner.AddInput(is_training_node[0], new TFTensor(false));

            foreach (var output in outputs)
            {
                var output_node = _graph[_prefix + "/" + output];
                runner.Fetch(output_node[0]);
            }

            // run and get and return results
            var eval_results = runner.Run();
            int output_idx = 0;
            foreach (var output in outputs)
            {
                result[output] = eval_results[output_idx++];
            }
            return result;
        }


        virtual public string ModelName
        {
            get
            {
                if (_params != null)
                {
                    string model_name = _params["model_name"].ToString();
                    var rev = _params["release_num"];
                    if (rev != null)
                        model_name += "." + rev.ToString();
                    return model_name;
                }
                else
                    return "";
            }
        }
    }
}
