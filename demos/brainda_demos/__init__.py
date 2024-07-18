import os


class ModelFactory:
    module_names = {
        0: 'DCPM',
        1: 'deep_learning',
        2: 'FBCSP',
        3: 'FBeCCA',
        4: 'FBMsCCA',
        5: 'FBmultiCSP',
        6: 'FBTDCA',
        7: 'FBTRCA',
        8: 'LDA',
        9: 'LST',
        10: 'MDRM',
        11: 'MEKT',
        12: 'msSAME_eTRCA',
        13: 'msSAME_TDCA',
        14: 'P300_demo',
        15: 'performance_demo',
        16: 'pretraining',
        17: 'RPA',
        18: 'SAME',
        19: 'sceTRCA',
        20: 'ssvep_demo',
    }

    @staticmethod
    def get_model(model_name, config):
        if model_name not in ModelFactory.module_names:
            raise ValueError(f"Invalid model name: {model_name}")

        module_name = ModelFactory.module_names[model_name]

        # 动态导入模块
        try:
            module = __import__(module_name, fromlist=[module_name])
        except ImportError as e:
            raise ImportError(f"Failed to import module {module_name}: {e}")

        # 获取模块内的特定类
        model_class = getattr(module, model_name, None)

        if model_class is None:
            raise AttributeError(f"No class found for model {model_name} in module {module_name}")

        # 实例化模型
        model_instance = model_class(**config)

        return model_instance



# 下面是如何使用这个工厂类的例子
if __name__ == '__main__':
    config = {
        'num_channels': 64,  # 假设的通道数
        'num_sample_points': 1024,  # 假设的样本点数
        'num_classes': 2,  # 假设的类别数
    }
    # 假设我们想要加载名为'deep_learning'的模型
    modle_name = '1'
    model = ModelFactory.get_model(modle_name, config)
    results = model.run()
    print(results)