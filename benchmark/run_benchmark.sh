for qps in 2 4 6 8 10 12
do
  python benchmark_serving.py --dataset /home/user/datasets/ShareGPT_V3_unfiltered_cleaned_split_1100.json --model Qwen --tokenizer /home/user/src/infini_test/public/models/Qwen_Qwen1.5-72B-Chat --base-url http://if-c7rqbnzdrq6bwr3j-service:80 --request-rate ${qps} --trust-remote-code --num-prompts 1000 --backend infini > /home/user/logs/infini/qps_${qps}.log
done

for qps in 2 4 6 8 10 12
do
  python benchmark_serving.py --dataset /home/user/datasets/ShareGPT_V3_unfiltered_cleaned_split_1100.json --model Qwen --tokenizer /home/user/src/infini_test/public/models/Qwen_Qwen1.5-72B-Chat --base-url http://if-c7ruvmypvcivgq63-service:80 --request-rate ${qps} --trust-remote-code --num-prompts 1000 --backend vllm > /home/user/logs/vllm/qps_${qps}.log
done
