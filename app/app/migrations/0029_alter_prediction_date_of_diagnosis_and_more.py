# Generated by Django 4.2.2 on 2025-03-30 02:31

import datetime
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0028_alter_prediction_date_of_diagnosis'),
    ]

    operations = [
        migrations.AlterField(
            model_name='prediction',
            name='date_of_diagnosis',
            field=models.DateTimeField(default=datetime.datetime(2025, 3, 30, 10, 31, 53, 214184), null=True),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='predicted_diesease',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.DO_NOTHING, to='app.disease'),
        ),
    ]
