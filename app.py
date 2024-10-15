import argparse
import logging
from typing import Union

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np


class GOTOCRWrapper(ClamsApp):

    def __init__(self):
        super().__init__()
        gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if gpu else 'cpu')
        self.logger.debug("Initializing GOT-OCR2_0 model")
        self.tokenizer = AutoTokenizer.from_pretrained(
            'ucaslcl/GOT-OCR2_0',
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            'ucaslcl/GOT-OCR2_0',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='auto' if gpu else None,
            use_safetensors=True,
            pad_token_id=self.tokenizer.eos_token_id
        ).to(self.device).eval()
        self.logger.debug("GOT-OCR2_0 model loaded successfully")

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory.
        # When using the ``metadata.py`` leave this do-nothing "pass" method here.
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        self.logger.debug("Running GOT-OCR Wrapper")
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view = mmif.get_views_for_document(video_doc.properties.id)[-1]
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)

        for timeframe in input_view.get_annotations(AnnotationTypes.TimeFrame):
            self.logger.debug(f"Processing TimeFrame: {timeframe.properties}")
            representatives = timeframe.get("representatives") if "representatives" in timeframe.properties else None
            if representatives:
                frame_number = vdh.get_representative_framenum(mmif, timeframe)
                image = vdh.extract_representative_frame(mmif, timeframe, as_PIL=True)
            else:
                frame_number = vdh.get_mid_framenum(mmif, timeframe)
                image = vdh.extract_mid_frame(mmif, timeframe, as_PIL=True)

            self.logger.debug("Extracted image for OCR")
            # Save the image to a temporary file
            image_file = 'temp_image.jpg'
            image.save(image_file)

            self.logger.debug("Running GOT-OCR2_0")
            with torch.no_grad():
                res = self.model.chat(self.tokenizer, image_file, ocr_type='ocr')
            
            self.logger.debug(f"OCR Results: {res}")
            timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
            timepoint.add_property('timePoint', frame_number)
            text_document = new_view.new_textdocument(res)
            alignment = new_view.new_annotation(AnnotationTypes.Alignment)
            alignment.add_property("source", timepoint.id)
            alignment.add_property("target", text_document.id)

        return mmif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = GOTOCRWrapper()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
