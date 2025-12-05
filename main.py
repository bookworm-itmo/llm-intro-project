from services.data_service.data_preparator import DataPreparator

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def main():
    """Главная функция для подготовки данных."""
    preparator = DataPreparator(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    preparator.prepare_all()


if __name__ == "__main__":
    main()
