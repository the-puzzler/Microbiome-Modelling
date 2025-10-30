#%% First we need to read in the data.
import csv
import os
import sys

# Make sure we can import model.py from repo root when running this script directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

run_rows = {}
SRA_to_micro = {}
gid_to_sample = {}
micro_to_subject = {}
micro_to_sample = {}


for path in ('data/diabimmune/SraRunTable_wgs.csv', 'data/diabimmune/SraRunTable_extra.csv'):
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row['Run'].strip()
            library_name = row.get('Library Name', '').strip()
            subject_id = row.get('host_subject_id', row.get('host_subject_id (run)', '')).strip()
            run_rows[run_id] = {'library': library_name, 'subject': subject_id}


with open('data/diabimmune/microbeatlas_samples.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        srs = row['#sid']
        rids = row['rids']
        if not rids:
            continue
        for rid in rids.replace(';', ',').split(','):
            rid = rid.strip()
            if rid:
                SRA_to_micro[rid] = srs


with open('data/diabimmune/samples.csv') as f:
    header = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        pieces = line.split(',')
        if header is None:
            header = pieces
            header[0] = header[0].lstrip('\ufeff')
            continue
        row = dict(zip(header, pieces))
        for key in ('gid_wgs', 'gid_16s'):
            gid = row.get(key, '').strip()
            if gid:
                gid_to_sample[gid] = {'subject': row['subjectID'], 'sample': row['sampleID']}


for run_id, srs in SRA_to_micro.items():
    run_info = run_rows.get(run_id, {})
    library_name = run_info.get('library', '')
    if library_name and library_name in gid_to_sample:
        micro_to_sample[srs] = gid_to_sample[library_name]
        micro_to_subject[srs] = gid_to_sample[library_name]['subject']
    elif run_info.get('subject'):
        micro_to_subject[srs] = run_info['subject']
# %%


"""
micro_to_subject now gives SRS -> subjectID so we can grab any per-subject sheet.
micro_to_sample hangs sampleID alongside that subjectID when the library matches.
SRA_to_micro links the raw SRR run IDs back to the microbeatlas SRS labels.
run_rows and gid_to_sample are just helpers to make those bridges work.
Once we have the subject IDs it is easy to join pregnancy_birth, diabetes, growth, etc.
"""


#%% Collect OTUs per microbeatlas sample.
import h5py
biom_path = 'data/microbeatlas/samples-otus.97.metag.minfilter.minCov90.noMulticell.rod2025companion.biom'

micro_to_otus = {}
needed_srs = set(SRA_to_micro.values()) | set(micro_to_subject.keys())

with h5py.File(biom_path) as biom_file:
    observation_names = [
        oid.decode('utf-8') if isinstance(oid, bytes) else str(oid)
        for oid in biom_file['observation/ids'][:]
    ]
    sample_ids = biom_file['sample/ids']
    sample_indices = biom_file['sample/matrix/indices']
    sample_indptr = biom_file['sample/matrix/indptr'][:]
    for idx in range(len(sample_ids)):
        sample_entry = sample_ids[idx]
        sample_entry = sample_entry.decode('utf-8') if isinstance(sample_entry, bytes) else str(sample_entry)
        sample_key = sample_entry.split('.')[-1]
        if sample_key not in needed_srs:
            continue
        start = sample_indptr[idx]
        end = sample_indptr[idx + 1]
        otu_list = [observation_names[i] for i in sample_indices[start:end]]
        micro_to_otus[sample_key] = otu_list
        if len(micro_to_otus) == len(needed_srs):
            break

print('otus mapped for', len(micro_to_otus), 'samples')
missing_srs = needed_srs - set(micro_to_otus)
if missing_srs:
    print('missing from biom:', len(missing_srs))
if micro_to_otus:
    example_key = next(iter(micro_to_otus))
    print('example', example_key, 'otus:', micro_to_otus[example_key][:5])


"""
Okay so now we have a link from microbe atlas ids to otus (bacteria)
We can now link the bacteria to their DNA embeddings from prokbert, run them through
the model, and get our fixed embeddings!
"""

#%%
import torch
from tqdm import tqdm
from model import MicrobiomeTransformer

checkpoint_path = 'data/model/checkpoint_epoch_0_final_epoch3_conf00.pt'

D_MODEL = 100
NHEAD = 5
NUM_LAYERS = 5
DROPOUT = 0.1
DIM_FF = 400
OTU_EMB = 384
TXT_EMB = 1536

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['model_state_dict']

model = MicrobiomeTransformer(
    input_dim_type1=OTU_EMB,
    input_dim_type2=TXT_EMB,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FF,
    dropout=DROPOUT
)

model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

print('model ready on', device)

#%%
prokbert_path = 'data/model/prokbert_embeddings.h5'

with h5py.File(prokbert_path) as emb_file:
    embedding_group = emb_file['embeddings']
    example_ids = []
    for key in embedding_group.keys():
        example_ids.append(key)
        if len(example_ids) == 10:
            break
    print('total prokbert embeddings:', len(embedding_group))
    print('example embedding ids:', example_ids)

#%%
sample_embeddings = {}
missing_otus = 0

with h5py.File(prokbert_path) as emb_file:
    embedding_group = emb_file['embeddings']
    for sample_key, otu_list in tqdm(micro_to_otus.items(), desc='Embedding samples'):
        otu_vectors = []
        for otu_id in otu_list:
            if otu_id in embedding_group:
                vec = embedding_group[otu_id][()]
                otu_vectors.append(torch.tensor(vec, dtype=torch.float32, device=device))
            else:
                missing_otus += 1
        if not otu_vectors:
            continue
        otu_tensor = torch.stack(otu_vectors, dim=0).unsqueeze(0)
        type2_tensor = torch.zeros((1, 0, TXT_EMB), dtype=torch.float32, device=device)
        mask = torch.ones((1, otu_tensor.shape[1]), dtype=torch.bool, device=device)
        with torch.no_grad():
            hidden_type1 = model.input_projection_type1(otu_tensor)
            hidden_type2 = model.input_projection_type2(type2_tensor)
            combined_hidden = torch.cat([hidden_type1, hidden_type2], dim=1)
            hidden = model.transformer(combined_hidden, src_key_padding_mask=~mask)
            sample_vec = hidden.mean(dim=1).squeeze(0).cpu()
        sample_embeddings[sample_key] = sample_vec

print('sample embeddings ready:', len(sample_embeddings))
print('missing otu embeddings:', missing_otus)
if sample_embeddings:
    first_key = next(iter(sample_embeddings))
    print('example sample embedding', first_key, sample_embeddings[first_key][:5])

#%%
