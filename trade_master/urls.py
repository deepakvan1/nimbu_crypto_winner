
from django.urls import path
from . import views

urlpatterns = [
    path('', views.analytics_page, name='analytics'),
    path('api/trade-analytics/<str:coin_pair>/', views.TradeAnalyticsView.as_view(), name='trade-analytics'),
    path('api/trade-analytics/', views.TradeAnalyticsView.as_view(), name='coin-pairs-list'),
    path('api/account/', views.account_details, name='account-api'),
]


import threading
from .views import bot
threading.Thread(target=bot).start()