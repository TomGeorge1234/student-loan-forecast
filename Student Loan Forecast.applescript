on run
    try
        do shell script "/bin/zsh '/Users/tomgeorge/projects/student-loan-forecast/launch_forecast.sh' </dev/null"
    on error errorMessage
        display alert "Student Loan Forecast could not open" message errorMessage as critical
    end try
end run
