import pandas as pd

def read_recall_column(csv_file_path):
    """
    지정된 CSV 파일에서 'Recall' 컬럼을 읽어와 출력합니다.

    Args:
        csv_file_path (str): 읽어올 CSV 파일의 경로입니다.
    """
    try:
        # CSV 파일을 pandas DataFrame으로 읽어옵니다.
        df = pd.read_csv(csv_file_path)

        # 'Recall' 컬럼이 파일에 있는지 확인합니다.
        if 'Recall' in df.columns:
            # 'Recall' 컬럼을 선택하여 출력합니다.
            recall_data = df['Recall']
            print("\n'Recall' 컬럼 데이터:")
            print(recall_data)
        else:
            print(f"오류: '{csv_file_path}' 파일에 'Recall' 컬럼이 없습니다.")

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {csv_file_path}")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    # 읽어올 CSV 파일 경로 (사용자로부터 받은 경로)
    csv_path = "D:\\AutoSigGen_withData\\Dataset_Paral\\signature\\NSL-KDD\\NSL-KDD_RARM_1_confidence_signature_train_ea15.csv"
    read_recall_column(csv_path)
