from datasets import Dataset
from vlm2vec_for_pyserini.data.eval_dataset.vidore_dataset import process_input_text, RESOLUTION_MAPPING, ImageVideoInstance
from vlm2vec_for_pyserini.data.eval_dataset.visrag_dataset import TASK_INST_QRY, TASK_INST_TGT


class VisDocDatasetForPyserini:
    def topics_dataset(batch_dict, **kwargs):
        model_backbone =  kwargs['model_backbone']
        query_texts, query_images, dataset_infos = [], [], []
        for query_id, query in zip(batch_dict['qid'], batch_dict['query']):
            query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=query)])
            query_images.append([None])
            dataset_infos.append({"qids": [query_id]})
        return Dataset.from_dict({"query_text": query_texts, "query_image": query_images, "dataset_infos": dataset_infos})

    def corpus_dataset(batch_dict, **kwargs):
        image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
        cand_texts, cand_images, dataset_infos = [], [], []
        for corpus_id, image_path in zip(batch_dict['corpus_id'], batch_dict['image_path']):
            cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True)])
            cand_images.append([ImageVideoInstance(
                bytes=[None],
                paths=[image_path],
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
            ).to_dict()])
            dataset_infos.append({
                "cand_name": [corpus_id],
            })

        return Dataset.from_dict({"cand_text": cand_texts, "cand_image": cand_images,
                "dataset_infos": dataset_infos})