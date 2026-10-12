#!/bin/zsh

# Double-click the desktop app to start the forecaster and open Safari.
cd "${0:A:h}" || exit 1
url='http://127.0.0.1:8502'
mkdir -p data || exit 1
log="$PWD/data/streamlit.log"

if ! /usr/bin/curl --silent --fail "$url/_stcore/health" >/dev/null; then
    if [[ ! -x "$PWD/.venv/bin/python" ]]; then
        print -u2 'Missing virtual environment. Follow the setup instructions in README.md.'
        exit 1
    fi
    /usr/bin/nohup "$PWD/.venv/bin/python" -m streamlit run student_loan_gui.py \
        --server.address 127.0.0.1 --server.port 8502 \
        --server.headless true --browser.gatherUsageStats false \
        >"$log" 2>&1 </dev/null &
    server_pid=$!

    for attempt in {1..60}; do
        if /usr/bin/curl --silent --fail "$url/_stcore/health" >/dev/null; then
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            print -u2 'The forecaster could not start. Details:'
            /bin/cat "$log" >&2
            exit 1
        fi
        /bin/sleep 1
    done
fi

if /usr/bin/curl --silent --fail "$url/_stcore/health" >/dev/null; then
    /usr/bin/open -a Safari "$url"
else
    print -u2 "The forecaster did not start in time. See $log for details."
    exit 1
fi
