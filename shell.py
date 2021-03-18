import pandas as pd
from user.models import DrawingExercise

def upload_to_db(row, fields, model):
    row = row.to_list()
    model.objects.create(**dict(zip(fields, row)))


GLOBAL_PATH = "https://raw.githubusercontent.com/kaavish-ki-kavish/aangan-filesystem/main/aangan-filesystem/"
def drawing_exercises_to_db(file_path):
    """ Handles reading lines from a file and saving to the Database.

    :param file_path: Path to where file is located.
    :type file_path: str
    """

    data_file = pd.read_csv(file_path, header = 0)
    data_file['stroke_path'] = GLOBAL_PATH + data_file['stroke_path']
    data_file['image_path'] = GLOBAL_PATH + data_file['image_path']
    print(f'Starting to upload {len(data_file.index)} records to Drawing Table')
    fields = ['label', 'ref_img_path', 'ref_stroke_path', 'level', 'sound_path']
    model = DrawingExercise
    kwargs = {'fields': fields, 'model': model}
    data_file.apply(upload_to_db, **kwargs, axis = 1)
    print(f'Starting to upload {len(data_file.index)} records to Drawing Table... DONE.')


# file_path = 'file_dir.csv'
# drawing_exercises_to_db(file_path)