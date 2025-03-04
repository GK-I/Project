import os
from flask_sqlalchemy import SQLAlchemy

def scan_and_update_videos(app):
    """Scan venue directories and update the ExamVideo table."""
    print("RUNNING VIDEO SCANNER")
    with app.app_context():  # Ensure correct app context
        from app import db, ExamVenue, ExamVideo  # Import inside function to avoid circular import

        venues = {venue.venue_name: venue.venue_id for venue in ExamVenue.query.all()}

        for venue_name, venue_id in venues.items():
            venue_path = os.path.normpath(os.path.join(app.config.get("VIDEO_BASE_DIR", ""), venue_name))

            if os.path.exists(venue_path):
                print(f"Directory exists: {venue_path}")
                print("Files in directory:", os.listdir(venue_path))  # Print all files
            else:
                print(f"Directory does NOT exist: {venue_path}")

            if os.path.exists(venue_path):
                existing_videos = {v.video_name for v in ExamVideo.query.filter_by(venue_id=venue_id).all()}
                for file in os.listdir(venue_path):
                    if file.endswith(('.mp4', '.avi', '.mov')) and file not in existing_videos:
                        print("Found videos:", file)
                        video_path = os.path.join(venue_path, file)

                        # Add new video entry
                        new_video = ExamVideo(
                            venue_id=venue_id,
                            video_name=file,
                            video_path=video_path,
                            processed=False
                        )
                        db.session.add(new_video)
                        print(f"Added: {file} to venue {venue_name}")

                db.session.commit()
