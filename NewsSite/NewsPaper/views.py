from datetime import datetime
from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from pygooglenews import GoogleNews
import requests



def has_internet_connection():
    try:
        requests.get("https://google.com", timeout=5)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False



def News(request):
    if has_internet_connection():
        gn = GoogleNews(lang='en', country='US')
        top = gn.top_news()
        top_entries = top['entries']
        entry_details = []

        for idx, entry in enumerate(top_entries, start=1):
        # Extracting publisher information from sub_articles
            publisher_name = entry['sub_articles'][0].get('publisher', '') if entry.get('sub_articles') else ''

            published_date_str = entry['published']
            published_date=datetime.strptime(published_date_str, "%a, %d %b %Y %H:%M:%S %Z"),
        
            entry_detail = {
                'index': idx,
                'title': entry['title'],
                'published_date': published_date,
                'news_summary': entry['summary'],
                'sub_articles': entry.get('sub_articles', []),
                'publisher': publisher_name,
            }
            entry_details.append(entry_detail)

    # Pagination
        paginator = Paginator(entry_details, 5)
        page = request.GET.get('page')

        try:
            entry_details = paginator.page(page)
        except PageNotAnInteger:
            entry_details = paginator.page(1)
        except EmptyPage:
            entry_details = paginator.page(paginator.num_pages)

        context = {'entry_details': entry_details}
        return render(request, "NewsPaper/newsindex.html", context)
    else:
      
         return render(request, "NewsPaper/noInternet.html")