from datetime import datetime, timezone


def create_timestamped_filename(base_name, extension="txt", timestamp=""):
    # Get the current date and time object
    now = datetime.now(timezone.utc)

    # Format the datetime object into a clean string
    if timestamp == "":
        timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Combine the base name and the timestamp
    full_filename = f"{base_name}_{timestamp}.{extension}"

    return full_filename
