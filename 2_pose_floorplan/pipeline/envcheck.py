"""Check the main environment (CoVisPose step + CPU steps).   python -m pipeline.envcheck"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import common as C  # noqa: E402

CKPT_SHA = {C.CHECKPOINT_FULL: 'd2daeb2af602a61c0c0d610c12d1f721fcc2b578fd59a45252cb295095775bed',
            C.CHECKPOINT_EMA: 'f87c312291516e46657d972ad9563278f38108a52c47181f1b31ca646e22436c'}
HN_SHA = 'bdef2e337133f8f2937f99e7643cbef7be56a745a1fe3dad00d855bdcadd0c96'


def main():
    ok = True
    print('python', sys.version.split()[0], sys.executable)
    for mod in ('numpy', 'PIL', 'shapely', 'matplotlib', 'scipy', 'torch'):
        try:
            m = importlib.import_module(mod)
            print(f'  {mod:10s} {getattr(m, "__version__", "")}')
        except Exception as e:
            print(f'  {mod:10s} MISSING ({e})')
            ok = False
    try:
        import torch
        cuda = torch.cuda.is_available()
        print('  CUDA', cuda, torch.cuda.get_device_name(0) if cuda else '')
        if not cuda:      # not an error: step 2 falls back to the CPU (scripts/manual_room_groups_v2.py --device)
            print('  NOTE: no usable CUDA GPU; CoVisPose will run on the CPU (several seconds per pair; about 10 s on 2 cores).'
                  ' The GeoVerify scan (1_room_finder) still needs a GPU.')
        import covispose.evaluate  # noqa: F401
        import scripts.manual_room_groups_v2  # noqa: F401
        print('  covispose code OK')
    except Exception as e:
        print('  covispose import FAILED', e)
        ok = False
    for name, rel, want in (('CoVisPose checkpoint', C.CHECKPOINT, CKPT_SHA[C.CHECKPOINT]), ('HorizonNet checkpoint', C.HN_CHECKPOINT, HN_SHA)):
        p = C.local(rel)
        good = p.exists() and C.digest(rel) == want
        print(f'  {name}: {"OK" if good else "MISSING or CHANGED"} ({p})')
        if not p.exists():
            print('    download it from the GitHub release of this package into that folder (README section 0)')
        ok &= good
    print('MAIN ENV OK' if ok else 'MAIN ENV PROBLEM')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
