#!/usr/bin/env python3
"""
自动应用所有改进配置

运行方法：
    python apply_improvements.py
    
功能：
1. 备份原配置
2. 应用改进的 env_config_v2.py
3. 应用改进的 reward_shaper_v2.py  
4. 修改 port_env.py 的碰撞终止条件
5. 验证所有修改
"""

import os
import shutil
import re
from datetime import datetime


class ImprovementApplier:
    """改进应用器"""
    
    def __init__(self):
        self.backup_dir = f"./backups_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.changes_made = []
        self.errors = []
        
    def backup_file(self, filepath):
        """备份文件"""
        if not os.path.exists(filepath):
            return False
            
        # 创建备份目录
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # 保持目录结构
        backup_path = os.path.join(self.backup_dir, filepath)
        backup_dir = os.path.dirname(backup_path)
        os.makedirs(backup_dir, exist_ok=True)
        
        # 备份
        shutil.copy2(filepath, backup_path)
        print(f"  ✅ 已备份: {filepath} -> {backup_path}")
        return True
    
    def apply_env_config(self):
        """应用环境配置"""
        print("\n[1/3] 应用环境配置...")
        
        src = './env_config_v2.py'
        dst = './config/env_config.py'
        
        if not os.path.exists(src):
            self.errors.append(f"找不到源文件: {src}")
            return False
        
        # 备份原文件
        if os.path.exists(dst):
            self.backup_file(dst)
        
        # 复制新文件
        shutil.copy2(src, dst)
        self.changes_made.append("env_config.py: 应用改进配置")
        print(f"  ✅ 已应用: {dst}")
        
        # 验证
        with open(dst, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'MAX_COLLISIONS = 100' in content or "'collision': -10.0" in content:
                print(f"  ✅ 验证成功: 碰撞配置已更新")
                return True
            else:
                self.errors.append("env_config.py 验证失败")
                return False
    
    def apply_reward_shaper(self):
        """应用奖励塑形器"""
        print("\n[2/3] 应用奖励塑形器...")
        
        src = './reward_shaper_v2.py'
        dst = './environment/reward_shaper.py'
        
        if not os.path.exists(src):
            self.errors.append(f"找不到源文件: {src}")
            return False
        
        # 备份原文件
        if os.path.exists(dst):
            self.backup_file(dst)
        
        # 复制新文件
        shutil.copy2(src, dst)
        self.changes_made.append("reward_shaper.py: 应用改进奖励函数")
        print(f"  ✅ 已应用: {dst}")
        
        # 验证
        with open(dst, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'reaching_near_target' in content or 'acceleration_bonus' in content:
                print(f"  ✅ 验证成功: 新增奖励项已添加")
                return True
            else:
                self.errors.append("reward_shaper.py 验证失败")
                return False
    
    def apply_port_env_fix(self):
        """修改port_env.py的终止条件"""
        print("\n[3/3] 修改port_env.py终止条件...")
        
        filepath = './environment/port_env.py'
        
        if not os.path.exists(filepath):
            self.errors.append(f"找不到文件: {filepath}")
            return False
        
        # 备份
        self.backup_file(filepath)
        
        # 读取文件
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查当前状态
        if "> 100" in content and "episode_stats['collisions']" in content:
            print(f"  ℹ️  已经是改进版本，无需修改")
            return True
        
        # 执行替换
        original_content = content
        
        # 替换1：修改数字
        content = re.sub(
            r"(self\.episode_stats\['collisions'\]\s*>\s*)30",
            r"\1100",
            content
        )
        
        # 替换2：修改注释
        content = re.sub(
            r"从10增加到30|从30增加到100",
            "从30增加到100（改进版 v2）",
            content
        )
        
        # 检查是否成功修改
        if content == original_content:
            self.errors.append("port_env.py 未找到需要修改的代码")
            print(f"  ⚠️  未找到需要修改的代码，请手动检查")
            return False
        
        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.changes_made.append("port_env.py: 碰撞终止条件 30 -> 100")
        print(f"  ✅ 已修改: 碰撞终止条件 30 -> 100")
        
        # 验证
        with open(filepath, 'r', encoding='utf-8') as f:
            if "> 100" in f.read():
                print(f"  ✅ 验证成功: 终止条件已更新")
                return True
            else:
                self.errors.append("port_env.py 验证失败")
                return False
    
    def verify_changes(self):
        """验证所有更改"""
        print("\n" + "="*60)
        print("📋 验证更改")
        print("="*60)
        
        all_good = True
        
        # 检查env_config.py
        print("\n检查 config/env_config.py:")
        if os.path.exists('./config/env_config.py'):
            with open('./config/env_config.py', 'r', encoding='utf-8') as f:
                content = f.read()
                checks = [
                    ("碰撞惩罚 -10.0", "'collision': -10.0" in content),
                    ("任务完成奖励 300.0", "'task_completion': 300.0" in content),
                    ("新增25米奖励", "'reaching_near_target'" in content),
                    ("新增加速奖励", "'acceleration_bonus'" in content),
                ]
                for name, result in checks:
                    status = "✅" if result else "❌"
                    print(f"  {status} {name}")
                    if not result:
                        all_good = False
        else:
            print("  ❌ 文件不存在")
            all_good = False
        
        # 检查reward_shaper.py
        print("\n检查 environment/reward_shaper.py:")
        if os.path.exists('./environment/reward_shaper.py'):
            with open('./environment/reward_shaper.py', 'r', encoding='utf-8') as f:
                content = f.read()
                checks = [
                    ("25米奖励逻辑", "current_dist < 25.0" in content),
                    ("加速奖励逻辑", "acceleration_bonus" in content),
                    ("改进版标记", "改进版 v2" in content or "v2" in content),
                ]
                for name, result in checks:
                    status = "✅" if result else "❌"
                    print(f"  {status} {name}")
                    if not result:
                        all_good = False
        else:
            print("  ❌ 文件不存在")
            all_good = False
        
        # 检查port_env.py
        print("\n检查 environment/port_env.py:")
        if os.path.exists('./environment/port_env.py'):
            with open('./environment/port_env.py', 'r', encoding='utf-8') as f:
                content = f.read()
                checks = [
                    ("碰撞终止条件 100", "> 100" in content),
                ]
                for name, result in checks:
                    status = "✅" if result else "❌"
                    print(f"  {status} {name}")
                    if not result:
                        all_good = False
        else:
            print("  ❌ 文件不存在")
            all_good = False
        
        return all_good
    
    def print_summary(self):
        """打印总结"""
        print("\n" + "="*60)
        print("📊 应用总结")
        print("="*60)
        
        print(f"\n✅ 成功应用的更改：")
        for change in self.changes_made:
            print(f"  - {change}")
        
        if self.errors:
            print(f"\n❌ 错误：")
            for error in self.errors:
                print(f"  - {error}")
        
        print(f"\n💾 备份目录：{self.backup_dir}")
        
        print("\n" + "="*60)
        
        if not self.errors:
            print("✅ 所有改进已成功应用！")
            print("\n下一步：")
            print("  1. python train_medium.py")
            print("  2. tensorboard --logdir=./runs_medium")
            print("  3. 监控 Tasks Completed 指标")
        else:
            print("⚠️  部分改进应用失败，请检查错误信息")
            print("\n手动修改指南：")
            print("  1. cp env_config_v2.py config/env_config.py")
            print("  2. cp reward_shaper_v2.py environment/reward_shaper.py")
            print("  3. 编辑 environment/port_env.py，将碰撞阈值 30 改为 100")
        
        print("="*60)
    
    def run(self):
        """运行应用流程"""
        print("="*60)
        print("🚀 自动应用改进配置")
        print("="*60)
        print("\n基于调试评估的改进：")
        print("  1. 放宽碰撞终止条件：30次 -> 100次")
        print("  2. 减小碰撞惩罚：-20 -> -10")
        print("  3. 增强接近奖励：15米内从5.0增加到15.0")
        print("  4. 新增加速奖励：鼓励AGV积极移动")
        print("  5. 新增25米奖励：填补引导空白")
        
        # 执行应用
        success = True
        success &= self.apply_env_config()
        success &= self.apply_reward_shaper()
        success &= self.apply_port_env_fix()
        
        # 验证
        if success:
            success &= self.verify_changes()
        
        # 打印总结
        self.print_summary()
        
        return success


def main():
    """主函数"""
    applier = ImprovementApplier()
    success = applier.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
