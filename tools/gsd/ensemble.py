import json
import os
from collections import defaultdict

import fire


gsd_estimated = {
    'v2': '/data/public/rw/team-autolearn/aerial/gsd_estimation/results/v2_190410/',
    'v3': '/data/public/rw/team-autolearn/aerial/gsd_estimation/results/v3_190412/'
}


class GSDEnsemble:
    def ensemble(self, method='avg', settype='valid', versions=('v2', 'v3'), out='./gsd_ensemble.json'):
        assert method in ['avg', 'list']

        merged = defaultdict(lambda: {'label': [], 'prediction': []})
        for v in versions:
            with open(os.path.join(gsd_estimated[v], '%s.json' % settype), 'r') as f:
                gsds = json.load(f)

            for img_id, gsd in gsds.items():
                merged[img_id]['label'].append(gsd['label'])
                merged[img_id]['prediction'].append(gsd['prediction'])

        for img_id, gsd in merged.items():
            assert len(gsd) == len(versions)

        if method == 'avg':
            for img_id, gsd in merged.items():
                merged[img_id]['label'] = sum(merged[img_id]['label']) / len(merged[img_id]['label'])
                merged[img_id]['prediction'] = sum(merged[img_id]['prediction']) / len(merged[img_id]['prediction'])

        if out:
            with open(out, 'w') as f:
                json.dump(merged, f)


if __name__ == '__main__':
    """
    Usage : python tools/gsd/ensemble.py ensemble --versions v2,v3 --out /data/public/rw/team-autolearn/aerial/gsd_estimation/results/ensemble_v2+v3_avg/test.json --settype test
    """
    fire.Fire(GSDEnsemble)
