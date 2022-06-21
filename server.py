import os
import sys
import time
import cv2
from prometheus_client import start_http_server, Gauge
from analog_gauge_reader import GaugeOption, calibrate_gauge, capture_stream, prepare_image, process

prometheus_gauge = Gauge('analog_gauge', 'Readout of an analog gauge', [
                         'name', 'location', 'unit'])

metrics_port = int(os.environ.get('METRICS_PORT', '8000'))
output_dir = os.environ.get('OUTPUT_DIR', '/tmp')
stream_url = os.environ['STREAM_URL']
interval_seconds = float(os.environ.get('INTERVAL_SECONDS', '10'))

def serve(gauges: list[GaugeOption]):

    start_http_server(metrics_port)
    print(f'started server on port {metrics_port}')

    last_time = 0.0

    while True:
        print('taking picture')
        capt = capture_stream(stream_url)

        if capt is not None:
            for g in gauges:
                print(f'reading gauge {g.name}')
                val = process(capt, g)
                if val is None:
                    print(f'could not read {g.name}')
                    continue

                print(f'{g.name}: {round(val, 0):.0f}')
                prometheus_gauge.labels(
                    name=g.name,
                    location='' if g.location is None else g.location,
                    unit='' if g.unit is None else g.unit).set(val)

        elapsed = time.monotonic() - last_time
        if elapsed < interval_seconds:
            print(f'sleep for {interval_seconds - elapsed}')
            time.sleep(interval_seconds - elapsed)
        last_time = time.monotonic()


def calibrate(gauges: list[GaugeOption]):
    capt = capture_stream(stream_url)
    if capt is not None:
        for gauge in gauges:
            img = prepare_image(capt, gauge)
            calibrate_gauge(img, gauge.name, lambda i, n: cv2.imwrite(f'{output_dir}/{n}.jpg', i) )


def main():
    if len(sys.argv) < 2:
        print("Commands: 'serve', 'calibrate' 'once'")
        sys.exit(0)

    if not stream_url:
        print("please specify a stream url via env STREAM_URL")
        sys.exit(0)

    gauges = []
    if len(sys.argv) >= 3:
        with open(sys.argv[2], "r", encoding='utf-8') as read_file:
            gauges = GaugeOption.schema().loads(read_file.read(), many=True)

    command = sys.argv[1]
    match command:
        case 'serve':
            serve(gauges)
        case 'calibrate':
            calibrate(gauges)
        case _:
            print("Commands: 'serve', 'calibrate' 'once'")


if __name__ == '__main__':
    main()
