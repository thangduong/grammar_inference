using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Grammar
{
    class CritiqueBeamSearch : CritiqueModel
    {
        CritiqueModel _originalCritique;
        public CritiqueBeamSearch(CritiqueModel originalCritique)
        {
            _originalCritique = originalCritique;
        }
        public override List<Critique> Eval(string sentence)
        {
            throw new NotImplementedException();
        }
    }
}
