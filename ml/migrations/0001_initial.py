# Generated by Django 4.2.7 on 2023-11-17 15:31

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="FileModel",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("file", models.FileField(upload_to="")),
                ("create_date", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
