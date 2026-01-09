if __name__.startswith("tools.webapp."):
    default_app_config = "tools.webapp.django_app.apps.HMWebappConfig"
else:
    default_app_config = "django_app.apps.HMWebappConfig"
