from django.contrib.auth.models import User


class AutoLoginMiddleware:
    """Automatically log in as admin superuser for demo purposes."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.user.is_authenticated:
            try:
                admin_user = User.objects.filter(is_superuser=True).first()
                if admin_user:
                    request.user = admin_user
            except Exception:
                pass
        return self.get_response(request)
