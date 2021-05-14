import pandas as pd

from user.models import Clusters, ClusterFeature, Features, Characters, ObjectWord, DrawingExercise


def upload_to_db(row, fields, model):
    row = row.to_list()
    model.objects.create(**dict(zip(fields, row)))


def cluster_to_db(file_path):
    """ Handles reading lines from a file and saving to the Database.

    :param file_path: Path to where file is located.
    :type file_path: str
    """

    data_file = pd.read_csv(file_path, header=0)
    fields = ['cluster_id', 'cluster_name']
    model = Clusters
    kwargs = {'fields': fields, 'model': model}
    data_file.apply(upload_to_db, **kwargs, axis=1)
    print(f'Starting to upload {len(data_file.index)} records to Table... DONE.')
    c = 'populate_scripts/cluster.csv'


def feature_to_db(file_path):
    """ Handles reading lines from a file and saving to the Database.

    :param file_path: Path to where file is located.
    :type file_path: str
    """

    data_file = pd.read_csv(file_path, header=0)
    fields = ['feature_id', 'feature_name']
    model = Features
    kwargs = {'fields': fields, 'model': model}
    data_file.apply(upload_to_db, **kwargs, axis=1)
    print(f'Starting to upload {len(data_file.index)} records to Table... DONE.')


def cluster_feature_to_db(cluster_file, feature_file):
    cluster_features = {}
    cluster_features['alif'] = ['startUpDownVertical', 'endVerticalDown', 'alif']
    cluster_features['bey'] = ['startUpDownVertical', 'endVerticalUp', 'longHorR2L']
    cluster_features['jeem'] = ['longHorL2R', 'semiCircleU2D', 'jeem']
    cluster_features['daal'] = ['daal']
    cluster_features['ray'] = ['ray', 'startUpDownVertical', 'sharpEdge']
    cluster_features['seen'] = ['endSemiCircle', 'sharpEdge', 'seen']
    cluster_features['swad'] = ['endSemiCircle', 'startSwaad']
    cluster_features['twa'] = ['startUpDownVertical', 'downIntersection']
    cluster_features['ayn'] = ['startAien', 'semiCircleU2D']
    cluster_features['faa'] = ['longHorL2R', 'startLoopUp', 'endVerticalUp']
    cluster_features['qaaf'] = ['startLoopUp', 'endSemiCircle']
    cluster_features['kaaf'] = ['startUpDownVertical', 'longHorR2L', 'endVerticalUp']
    cluster_features['laam'] = ['startUpDownVertical', 'semiCircleR2L']
    cluster_features['noon'] = ['semiCircleR2L']
    cluster_features['meem'] = ['endVerticalDown', 'startLoopDown']
    cluster_features['waw'] = ['startLoopUp']
    cluster_features['gool-hay'] = ['goolHay']
    cluster_features['chashmi-ha'] = ['chashmiHay']
    cluster_features['choti-yay'] = ['endChotiYay', 'endSemiCircle']
    cluster_features['bari-yay'] = ['longHorL2R']

    cluster_df = pd.read_csv(cluster_file, header=0)
    feature_df = pd.read_csv(feature_file, header=0)
    # print(cluster_df.head(len(cluster_df)))

    # a = cluster_df.index[cluster_df['cluster_name'] == 'alif'].tolist()[0]
    #
    # b = cluster_df.loc[a, 'cluster_id']
    # print(b)

    #
    for key, value in cluster_features.items():
        key_row = cluster_df.index[cluster_df['cluster_name'] == key.strip(" ")].tolist()[0]
        cluster_id = cluster_df.loc[key_row, 'cluster_id']
        table_cluster_id = Clusters.objects.get(cluster_id=cluster_id)

        for i in value:
            value_row = feature_df.index[feature_df['feature_name'] == i.strip(" ")].tolist()[0]
            feature_id = feature_df.loc[value_row, 'feature_id']
            table_feature_id = Features.objects.get(feature_id=feature_id)
            ClusterFeature.objects.create(feature_id=table_feature_id, cluster_id=table_cluster_id)


def characters_to_db():
    urdu_csv_file = 'https://raw.githubusercontent.com/kaavish-ki-kavish/aangan-filesystem/main/aagan-urdu-filesystem/urdu_file_dir.csv'
    cluster_csv_file = 'populate_scripts/cluster.csv'
    GLOBAL_PATH = 'https://raw.githubusercontent.com/kaavish-ki-kavish/aangan-filesystem/main/aagan-urdu-filesystem/'

    clusters = {}
    clusters['alif'] = ['alif', 'alif-mad-aa']
    clusters['bey'] = ['ttaa', 'paa', 'seey', 'baa', 'taa']
    clusters['jeem'] = ['khaa', 'jeem', 'haa1', 'cheey']
    clusters['daal'] = ['daal', 'zaal', 'dhaal']
    clusters['ray'] = ['rhraa', 'raa', 'zhaa', 'zaaa']
    clusters['seen'] = ['seen', 'sheen']
    clusters['swad'] = ['zwaad', 'swaad']
    clusters['twa'] = ['twa', 'zwaa', 'Twaa']
    clusters['ayn'] = ['ayn', 'ghain']
    clusters['faa'] = ['faa']
    clusters['qaaf'] = ['qaaf']
    clusters['kaaf'] = ['gaaf', 'kaaf']
    clusters['laam'] = ['laam']
    clusters['meem'] = ['meem']
    clusters['noon'] = ['noon', 'noonghunna']
    clusters['waw'] = ['waw']
    clusters['gool-hay'] = ['haa3']
    clusters['chashmi-ha'] = ['haa2']
    clusters['choti-yay'] = ['choti-yaa']
    clusters['bari-yay'] = ['bari-yaa']

    sequence_letters = ['alif', 'alif-mad-aa', 'baa', 'paa', 'taa', 'ttaa', 'seey',
                        'jeem', 'cheey', 'haa1', 'khaa', 'daal', 'dhaal', 'zaal', 'raa',
                        'rhraa', 'zaaa', 'zhaa', 'seen', 'sheen', 'swaad', 'zwaad', 'Twaa',
                        'zwaa', 'ayn', 'ghain', 'faa', 'qaaf', 'kaaf', 'gaaf', 'laam', 'meem',
                        'noon', 'noonghunna', 'waw', 'haa2', 'haa3', 'choti-yaa', 'bari-yaa'
                        ]

    urdu_data = pd.read_csv(urdu_csv_file)
    cluster_data = pd.read_csv(cluster_csv_file)

    urdu_data['character_id'] = 0
    urdu_data['cluster_id'] = 1
    urdu_data['level'] = 0
    urdu_data['sequence_id'] = 0

    urdu_data.drop(urdu_data[urdu_data['label'] == 'Hamza'].index, inplace=True)
    urdu_data.drop(['object_sound_path', 'object_letter_image_path', ], axis=1, inplace=True)

    for key, item in clusters.items():
        item_cluster_id = cluster_data.loc[cluster_data['cluster_name'] == key, 'cluster_id'].values[0]
        for i in item:
            if i != 'twa':
                cluster_index = urdu_data.index[urdu_data['label'] == i].tolist()[0]
                urdu_data.at[cluster_index, 'cluster_id'] = Clusters.objects.get(cluster_id=item_cluster_id)
                urdu_data.at[cluster_index, 'sequence_id'] = sequence_letters.index(i) + 1
                urdu_data.at[cluster_index, 'character_id'] = sequence_letters.index(i)

    urdu_data.rename(columns={'letter_image_path': 'ref_stroke_path', 'letter_sound_path': 'sound_path',
                              'object_image_path': 'ref_object_path'}, inplace=True)

    urdu_data['ref_stroke_path'] = GLOBAL_PATH + urdu_data['ref_stroke_path']
    urdu_data['sound_path'] = GLOBAL_PATH + urdu_data['sound_path']
    urdu_data['ref_object_path'] = GLOBAL_PATH + urdu_data['ref_object_path']

    fields = ['label', 'ref_stroke_path', 'sound_path', 'ref_object_path', 'character_id', 'cluster_id', 'level',
              'sequence_id']
    model = Characters
    kwargs = {'fields': fields, 'model': model}
    urdu_data.apply(upload_to_db, **kwargs, axis=1)
    print(f'Starting to upload {len(urdu_data.index)} records to Table... DONE.')


def object_word_to_db():
    urdu_csv_file = 'https://raw.githubusercontent.com/kaavish-ki-kavish/aangan-filesystem/main/aagan-urdu-filesystem/urdu_file_dir.csv'
    GLOBAL_PATH = 'https://raw.githubusercontent.com/kaavish-ki-kavish/aangan-filesystem/main/aagan-urdu-filesystem/'
    urdu_data = pd.read_csv(urdu_csv_file)
    urdu_data.drop(urdu_data[urdu_data['label'] == 'Hamza'].index, inplace=True)
    urdu_data.drop(['letter_image_path', 'letter_sound_path', ], axis=1, inplace=True)
    urdu_data['is_object'] = False
    urdu_data['category'] = " "

    urdu_data.rename(columns={'object_image_path': 'image_path', 'object_sound_path': 'sound_path',
                              'object_letter_image_path': 'ref_image_path'}, inplace=True)

    urdu_data['image_path'] = GLOBAL_PATH + urdu_data['image_path']
    urdu_data['sound_path'] = GLOBAL_PATH + urdu_data['sound_path']
    urdu_data['ref_image_path'] = GLOBAL_PATH + urdu_data['ref_image_path']

    fields = ['label', 'image_path', 'sound_path', 'ref_image_path', 'is_object', 'category']

    model = ObjectWord
    kwargs = {'fields': fields, 'model': model}
    urdu_data.apply(upload_to_db, **kwargs, axis=1)
    print(f'Starting to upload {len(urdu_data.index)} records to Table... DONE.')


def drawing_exercises_to_db(file_path):
    GLOBAL_PATH = "https://raw.githubusercontent.com/kaavish-ki-kavish/aangan-filesystem/main/aangan-filesystem/"

    """ Handles reading lines from a file and saving to the Database.

    :param file_path: Path to where file is located.
    :type file_path: str
    """

    data_file = pd.read_csv(file_path, header=0)
    data_file['stroke_path'] = GLOBAL_PATH + data_file['stroke_path']
    data_file['image_path'] = GLOBAL_PATH + data_file['image_path']
    print(f'Starting to upload {len(data_file.index)} records to Drawing Table')
    fields = ['label', 'ref_img_path', 'ref_stroke_path', 'level', 'sound_path']
    model = DrawingExercise
    kwargs = {'fields': fields, 'model': model}
    data_file.apply(upload_to_db, **kwargs, axis=1)
    print(f'Starting to upload {len(data_file.index)} records to Drawing Table... DONE.')
