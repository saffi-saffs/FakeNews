from django.db import models

class GoogleNewsArticle(models.Model):
    title = models.CharField(max_length=800)
    link = models.URLField()
    published_date = models.DateTimeField()
    publisher = models.CharField(max_length=800, default='Unknown Publisher')  # Specify the default value here
   
    def __str__(self):
        return self.title
