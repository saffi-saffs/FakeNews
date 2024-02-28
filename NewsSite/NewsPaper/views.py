# views.py

from .models import GoogleNewsArticle
from datetime import datetime
from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from pygooglenews import GoogleNews

def News(request):
    # Fetch articles from Google News
    gn = GoogleNews(lang='en', country='US')
    top = gn.top_news()
    top_entries = top['entries']
    entry_details = []

    for idx, entry in enumerate(top_entries, start=1):
        # Extracting publisher information from sub_articles
        publisher_name = entry['sub_articles'][0].get('publisher', '') if entry.get('sub_articles') else ''

        published_date_str = entry['published']
        
        # Check if the article already exists in the database by checking its title
        if not GoogleNewsArticle.objects.filter(title=entry['title']).exists():
            # Create and save the article only if it doesn't already exist
            article = GoogleNewsArticle.objects.create(
                title=entry['title'],
                link=entry['link'],
                published_date=datetime.strptime(published_date_str, "%a, %d %b %Y %H:%M:%S %Z"),
                publisher=publisher_name,
            )

        entry_detail = {
            'index': idx,
            'title': entry['title'],
            'published_date': entry['published'],
            'news_summary': entry['summary'],
            'sub_articles': entry.get('sub_articles', []),
            'publisher': publisher_name,
        }
        entry_details.append(entry_detail)

    # Retrieve all articles from the database
    articles = GoogleNewsArticle.objects.all()
    
    # Paginate the articles
    paginator = Paginator(articles, 5)
    page = request.GET.get('page')

    try:
        articles = paginator.page(page)
    except PageNotAnInteger:
        articles = paginator.page(1)
    except EmptyPage:
        articles = paginator.page(paginator.num_pages)

    context = {'entry_details': entry_details}
    return render(request, "NewsPaper/newsindex.html", context)

# database portion
def google_news_database(request):
    articles = GoogleNewsArticle.objects.all()

    paginator = Paginator(articles, 5)
    page = request.GET.get('page')

    try:
        articles = paginator.page(page)
    except PageNotAnInteger:
        articles = paginator.page(1)
    except EmptyPage:
        articles = paginator.page(paginator.num_pages)

    context = {'articles': articles}
    return render(request, 'NewsPaper/todaysNews.html', context)

# database views.py


def show_articles(request):
    # Retrieve articles from the database
    articles = GoogleNewsArticle.objects.all()
    
    # Pass articles to the template context
    context = {'articles': articles}
    
    # Render the template with the provided context
    return render(request,'NewsPaper/todaysNews.html',context)
