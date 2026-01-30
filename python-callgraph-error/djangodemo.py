from django.db.models.functions import Trunc
from django.db.models.functions import Extract
from django.db import models
from django.db.models.functions import Extract
import os

class MyModel(models.Model):
    date_field = models.DateTimeField()

def upadteOject():
    queryset = MyModel.objects.annotate(month=Trunc('date_field', 'month'))
    queryset = MyModel.objects.annotate(year=Extract('year', 'date_field'))

if __name__ == '__main__':
    os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'
    upadteOject()
