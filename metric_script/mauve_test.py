import mauve 

p_text = [
    "一般的小数可以用简单筛法找出质数列表，然后一个个试。这种方法简单暴力，但是对几亿以下的数字可以很快。",
    "先降一维理解一下，在平面上的生物只能看到直线的一边."
]

q_text = [
    "Ordinary decimals can be found with a simple sieve to find a list of prime numbers, and then try one by one. This method is simple and violent, but can be fast on numbers below hundreds of millions.",
    "Let's go down one dimension and understand that a creature on a plane can only see one side of a straight line."
]

featurize_model_name = "/home/xyf/paper/huggingface/model/bert-base-multilingual-cased"

# call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to 256 tokens
out = mauve.compute_mauve(p_text=p_text, 
                          q_text=q_text, 
                          device_id=0, 
                          max_text_length=256, 
                          verbose=True,
                          featurize_model_name=featurize_model_name)
print(out.mauve) # prints 0.9917