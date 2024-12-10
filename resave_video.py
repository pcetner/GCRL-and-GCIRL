import cv2

def resave_video(input_path, output_path):
    # Open the input video file
    cap = cv2.VideoCapture(input_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open the video file {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4

    # Create VideoWriter object for the output file
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    print(f"Resaving video to {output_path}...")

    # Read frames and write them to the new file
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Exit loop if no more frames

        out.write(frame)  # Write the current frame to the output file

    # Release resources
    cap.release()
    out.release()
    print(f"Video successfully saved to {output_path}")

if __name__ == "__main__":
    # Path to the input MP4 file
    input_video_path = "verification.mp4"  # Change this if the file is not in the current directory

    # Path to save the output MP4 file
    output_video_path = "D:/last/verification_resaved.mp4"

    # Resave the video
    resave_video(input_video_path, output_video_path)
