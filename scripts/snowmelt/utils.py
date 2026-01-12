import csv
import os


def parse_isolate_field(val: str):
    """
    Parse isolate strings like 'Fungi_Block7_IceEncasement_Csampling'.
    Returns (block:int|None, treatment:str|None, time:str|None) where
    treatment in {'IE','AMB','NoICE','NoSNOW','COMP'} and time in {'A','B','C'}.
    """
    parts = [p for p in (val or '').strip().split('_') if p]
    block = None
    treatment = None
    time = None
    for p in parts:
        lp = p.lower()
        if lp.startswith('block'):
            try:
                block = int(''.join(ch for ch in p if ch.isdigit()))
            except Exception:
                block = None
        elif lp.endswith('sampling') and len(p) >= 1:
            time = p[0].upper()  # A/B/C
        elif lp in (
            'iceencasement',
            'ambient',
            'noice',
            'nosnow',
            'compactedsnow',
        ):
            treatment = {
                'iceencasement': 'IE',
                'ambient': 'AMB',
                'noice': 'NoICE',
                'nosnow': 'NoSNOW',
                'compactedsnow': 'COMP',
            }[lp]
    return block, treatment, time


def load_snowmelt_metadata(csv_path='data/snowmelt/snowmelt.csv'):
    """
    Load snowmelt metadata CSV and return:
      - run_meta: {Run -> {'block':int, 'treatment':str, 'time':str}}
      - run_to_srs: {Run -> SRS sample_acc}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    run_meta = {}
    run_to_srs = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row.get('Run', '').strip()
            srs = row.get('sample_acc', '').strip()
            if run and srs:
                run_to_srs[run] = srs
            block, treatment, time = parse_isolate_field(row.get('isolate', ''))
            if run and block is not None and treatment and time:
                run_meta[run] = {'block': block, 'treatment': treatment, 'time': time}
    return run_meta, run_to_srs
