"""ds_test dataset."""

import tensorflow_datasets as tfds
import os
from imageprocessing import imagestack
_DESCRIPTION = """\
Dataset provided by Dr Deschiantre.
"""

# TODO(ds_test): BibTeX citation
_CITATION = """\
Not availiable atm
"""


class DsTest(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ds_test dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release for reusablity',
  }

  def _info(self) -> tfds.core.DatasetInfo:

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'single_flashed_image': tfds.features.Image(shape=(256, 256, 3)),
            'SVBRDF': tfds.features.Image(shape=(256, 256, 9)),
        }),
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://team.inria.fr/graphdeco/fr/projects/deep-materials/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    #URLpath   = dl_manager.download_and_extract('https://repo-sam.inria.fr/fungraph/deep-materials/DeepMaterialsData.zip')
    #windows format
    Localpath = dl_manager.manual_dir / 'E:\workspace_ms_zhiyuan\Data_Deschaintre18\\'

    #No validation atm
    test_path  = os.path.join(Localpath, 'testBlended')
    train_path = os.path.join(Localpath,"trainBlended")


    return {
        'train': self._generate_examples(path = train_path),
        'test' : self._generate_examples(path =  test_path),
    }

  def _generate_examples(self, img_path):

    index = 0
    for file in img_path.glob('*.png'):
      photo, svbrdf = imagestack(file)
      record = {
        'photo': photo,
        'svbrdf':svbrdf
      }
      yield index, record
      index += 1
