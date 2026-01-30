# 1. Create a loop that checks if the training process exists
while pgrep -f "verl.trainer.main_ppo" > /dev/null; do
    echo "[$(date)] Training is still running... Checking again in 60s."
    sleep 300
done

# 2. Once the loop breaks (process not found), initiate shutdown
echo "Training finished! Shutting down in 60 seconds... (Press Ctrl+C to cancel)"
sleep 300
sudo shutdown -h now
