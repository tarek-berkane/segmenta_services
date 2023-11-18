from django.db import models


# Create your models here.
class FileModel(models.Model):
    file = models.FileField()

    create_date = models.DateTimeField(auto_now_add=True)

