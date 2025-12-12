from django.urls import include, path

urlpatterns = [
    path("", include("hmwebapp.webapp.urls")),
]

