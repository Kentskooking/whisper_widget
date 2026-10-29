"""Run the existing desktop supervisor with a Windows single-instance guard."""
from app.bark_runtime import SingleInstance


def main():
    instance = SingleInstance()
    try:
        from app.supervisor import main as supervise
        return supervise()
    finally:
        instance.close()


if __name__ == "__main__":
    raise SystemExit(main())
