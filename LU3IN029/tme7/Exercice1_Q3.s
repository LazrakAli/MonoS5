.data

ch:	.asciiz	"l exemple d'exemple\n"


.text

main:
# prologue
	addiu	$29,$29,-4
# int i=0
	sw	$0,0($29)
	lw	$9,0($29)

# printf("%s", ch);
	lui	$8,0x1001
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
	
	lui $8,0x1001
# while (ch[i] != ’\0’)
while:
	lb 	$10,0($8)
	beq	$0,$10,endwhile
	ori	$4,$10,0
# ch[i] = min_to_maj_char(ch[i]);
	jal min_to_maj_char
	ori	$10,$2,0
	sb	$10,0($8)
# i++;
	addiu	$8,$8,1
	addiu	$9,$9,1
	sw	$9,0($29)

	j while

endwhile:

# printf("%s", ch);
	lui	$8,0x1001
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
#epilogue
	addiu	$29,$29,4	

# exit()
	ori	$2,$0,10
	syscall

min_to_maj_char:
# prologue
	addiu	$29,$29,-8
	sw	$4,4($29)
	sw	$31,8($29)

	ori	$15,$0,0x20
# if (ch[i] >= ’a’ && ch[i] <= ’z’)
	ori	$12,$0,0x61
    	slt     $12, $4, $12
    	bgtz    $12, endif
		
	ori	$12,$0,0x7A
    	slt     $12, $4,$4
    	bgtz    $12, endif
#ch[i] = ch[i] - 0x20;
	sub	$15,$4,$15
	ori	$2,$15,0
# epilogue	
	addiu	$29,$29,8
	jr $31
endif:
# epilogue
	addiu	$29,$29,8
	ori	$2,$4,0
	jr $31
	


	