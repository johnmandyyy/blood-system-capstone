# Generated by Django 4.2.2 on 2025-03-25 17:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0019_images_used_for'),
    ]

    operations = [
        migrations.AddField(
            model_name='modelinfo',
            name='json_info',
            field=models.TextField(default=None, null=True),
        ),
    ]
