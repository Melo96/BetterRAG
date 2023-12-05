import json
from tqdm import tqdm
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

# Get elements
raw_pdf_list = []
data_path = Path("/data/kaiwen/data/知识库/硬件")
pdf_files = list(data_path.rglob("*.pdf"))

for p in tqdm(pdf_files):
    raw_pdf_elements = partition_pdf(filename=p,
                                     include_metadata=True,
                                     include_page_breaks=False,
                                     # Unstructured first finds embedded image blocks
                                     extract_images_in_pdf=True,
                                     languages=["chi_sim"],
                                     image_output_dir_path="/data/kaiwen/knowledge-agent/data/images/customer_suport",
                                     # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
                                     # Titles are any sub-section of the document
                                     strategy='hi_res',
                                     infer_table_structure=True,
                                    )
    raw_pdf_list.append((str(p.relative_to(data_path)), raw_pdf_elements))

chunk_list = []
for file_name, raw_pdf_elements in tqdm(raw_pdf_list):
    chunks = chunk_by_title(raw_pdf_elements,
                            combine_text_under_n_chars=1000,
                            max_characters=2000,
                            new_after_n_chars=2500,
                            )
    chunk_list.append((file_name, chunks))

save_path = Path("/data/kaiwen/knowledge-agent/data/pdf_partition/customer_support/硬件")
save_path.mkdir(parents=True, exist_ok=True)
for file_name, raw_pdf_elements in tqdm(raw_pdf_list):
    cur_save_path = save_path / ("/".join(file_name.split("/")[:-1]))
    cur_save_path.mkdir(parents=True, exist_ok=True)
    save_file = save_path / (file_name + '.json') 
    save_list = [{"text": element.text, "type": element.category, "id": element.id, "metadata": element.metadata.to_dict()} for element in raw_pdf_elements]
    
    with open(save_file, 'w') as f:
        json.dump(save_list, f)

save_path = Path("/data/kaiwen/knowledge-agent/data/chunks/customer_support/硬件")
save_path.mkdir(parents=True, exist_ok=True)
for file_name, chunks in tqdm(chunk_list):
    cur_save_path = save_path / ("/".join(file_name.split("/")[:-1]))
    cur_save_path.mkdir(parents=True, exist_ok=True)
    save_file = save_path / (file_name + '.json') 
    save_list = [{"text": element.text, "type": element.category, "id": element.id, "metadata": element.metadata.to_dict()} for element in chunks]

    with open(save_file, 'w') as f:
        json.dump(save_list, f)