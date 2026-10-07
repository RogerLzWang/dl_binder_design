import argparse
from Bio.PDB import PDBIO, MMCIFParser
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

parser = argparse.ArgumentParser(description = "Convert CIF files to PDB.")
parser.add_argument("input_file", help="Path to the input mmCIF file")
parser.add_argument("output_file", help="Path to the output PDB file")

def convert_cif_to_pdb(cif_file, pdb_file):
    """Converts a mmCIF file to PDB format.

    Args:
        cif_file (str): Path to the input mmCIF file.
        pdb_file (str): Path to the output PDB file.
    """
    parser = MMCIFParser()
    try:
        structure = parser.get_structure("structure", cif_file)
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_file)
    except Exception as e:
        print(f"Error during conversion: {e}")


if __name__ == "__main__":
    args = parser.parse_args()
    cif_file_path = args.input_file
    pdb_file_path = args.output_file
    convert_cif_to_pdb(cif_file_path, pdb_file_path)
