# Generated by Django 3.1.5 on 2021-02-09 11:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0004_auto_20201205_2117'),
    ]

    operations = [
        migrations.AlterField(
            model_name='childprofile',
            name='level',
            field=models.IntegerField(choices=[(0, 'Montessori'), (1, 'Nursury'), (2, 'Prep'), (3, 'other')]),
        ),
    ]
