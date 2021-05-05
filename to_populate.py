import pandas as pd


from user.models import Clusters, ClusterFeature, Features


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
        table_cluster_id = Clusters.objects.get(cluster_id= cluster_id)

        for i in value:
            value_row = feature_df.index[feature_df['feature_name'] == i.strip(" ")].tolist()[0]
            feature_id = feature_df.loc[value_row, 'feature_id']
            table_feature_id = Features.objects.get(feature_id= feature_id)
            ClusterFeature.objects.create(feature_id= table_feature_id, cluster_id = table_cluster_id)



# cluster_file = 'populate_scripts/cluster.csv'
# feature_file = 'populate_scripts/features.csv'
# cluster_feature_to_db(cluster_file, feature_file)
