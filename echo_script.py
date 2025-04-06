import sys

if __name__ == "__main__":
    # Optional: Print a startup message to stderr to confirm execution
    print("Echo script started.", file=sys.stderr)
    sys.stderr.flush()
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                # End of input stream
                print("Input stream closed. Exiting echo script.", file=sys.stderr)
                break
            # Echo the received line (with potentially added newline) to stdout
            print(line, end='') # end='' because readline usually includes newline
            sys.stdout.flush() # Ensure output is sent immediately
            # Optional: Log received line to stderr
            # print(f"Echo script received: {line.strip()}", file=sys.stderr)
            # sys.stderr.flush()
    except KeyboardInterrupt:
        print("Echo script interrupted.", file=sys.stderr)
    except Exception as e:
        print(f"Echo script error: {e}", file=sys.stderr)
    finally:
        print("Echo script finished.", file=sys.stderr)
        sys.stderr.flush() 