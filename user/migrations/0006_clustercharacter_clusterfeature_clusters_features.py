# Generated by Django 3.1.5 on 2021-04-14 11:14

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0005_auto_20210209_1622'),
    ]

    operations = [
        migrations.CreateModel(
            name='Clusters',
            fields=[
                ('cluster_id', models.AutoField(primary_key=True, serialize=False)),
                ('cluster_name', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='Features',
            fields=[
                ('feature_id', models.AutoField(primary_key=True, serialize=False)),
                ('feature_name', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='ClusterFeature',
            fields=[
                ('cluster_feature_id', models.AutoField(primary_key=True, serialize=False)),
                ('cluster_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='user.clusters')),
                ('feature_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='user.features')),
            ],
        ),
        migrations.CreateModel(
            name='ClusterCharacter',
            fields=[
                ('cluster_character_id', models.AutoField(primary_key=True, serialize=False)),
                ('character_id', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='user.characters')),
                ('cluster_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='user.clusters')),
            ],
        ),
    ]
