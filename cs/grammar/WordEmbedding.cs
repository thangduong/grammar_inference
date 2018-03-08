using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Grammar
{
    public class WordEmbedding
    {
        Dictionary<string, float[]> _embedding = new Dictionary<string, float[]>();
        public void LoadText(string filename, bool writeBinFile = false)
        {
            string line;
            using (System.IO.StreamReader file =
                new System.IO.StreamReader(filename))
            {

                BinaryWriter writer = null;
                if (writeBinFile)
                {
                    writer = new BinaryWriter(File.Open(filename + ".bin", FileMode.Create));
                    if (writer != null)
                        writer.Write(300);
                }
                while ((line = file.ReadLine()) != null)
                {
                    string[] pieces = line.Split(null);
                    float[] vector = new float[pieces.Length - 1];
                    if (writer != null)
                        writer.Write(pieces[0]);
                    for (int j = 0; j < pieces.Length - 1; j++)
                    {
                        vector[j] = float.Parse(pieces[j + 1]);
                        if (writer != null)
                            writer.Write(vector[j]);
                    }
                    _embedding[pieces[0]] = vector;
                }
                if (writer != null)
                    writer.Close();
            }
        }
        public void LoadBinary(string filename)
        {
            using (BinaryReader reader = new BinaryReader(File.Open(filename, FileMode.Open)))
            {
                try
                {
                    int vectorSize = reader.ReadInt32();
                    while (true)
                    {
                        string word = reader.ReadString();
                        float[] vector = new float[vectorSize];
                        for (var i = 0; i < vectorSize; i++)
                            vector[i] = reader.ReadSingle();
                        _embedding[word] = vector;
                    }
                } catch (EndOfStreamException)
                {

                }
            }            
        }

        public float WordSimilarity(string w1, string w2)
        {
            float result;
            if (_embedding.ContainsKey(w1) && _embedding.ContainsKey(w2))
            {
                result = 0;
                float[] v1 = _embedding[w1];
                float[] v2 = _embedding[w2];
                float n1 = 0, n2 = 0;
                for (int i = 0; i < v1.Length; i++)
                {
                    result += v1[i] * v2[i];
                    n1 += v1[i] * v1[i];
                    n2 += v2[i] * v2[i];
                }
                result /= (float)Math.Sqrt(n1 * n2);
            }
            else
                result = -1;
            return result;
        }
    }
}
